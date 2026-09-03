// Compile and inspect representative policies at every static size budget.
#include "common/closed_form/concepts.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/monte_carlo/concepts.cuh"
#include "common/sample/sample_kernels.cuh"
#include "common/simulation/concepts.cuh"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

template<std::size_t Bytes>
struct Payload {
    std::uint8_t bytes[Bytes];
};

struct StateProvider {
    struct State {
        float value;
    };
};

template<std::size_t Bytes>
struct HandlerProbe {
    std::uint8_t bytes[Bytes];

    __device__ bool on_initial_state(const StateProvider::State&) {
        return bytes[0] == 0U;
    }

    __device__ bool on_observation(
        std::uint32_t,
        const StateProvider::State&
    ) {
        return bytes[Bytes - 1U] == 0U;
    }
};

template<std::size_t Bytes>
__global__ void thread_payload_kernel(
    const Payload<Bytes>* input,
    std::uint32_t* output
) {
    Payload<Bytes> local = input[0];
    std::uint32_t sum = 0U;
    for (std::size_t index = threadIdx.x; index < Bytes; index += blockDim.x) {
        sum += local.bytes[index];
    }
    if (threadIdx.x == 0U) output[0] = sum;
}

template<std::size_t Bytes>
__global__ void shared_payload_kernel(std::uint32_t* output) {
    __shared__ Payload<Bytes> payload;
    for (std::size_t index = threadIdx.x; index < Bytes; index += blockDim.x) {
        payload.bytes[index] = static_cast<std::uint8_t>(index);
    }
    __syncthreads();
    if (threadIdx.x == 0U) output[0] = payload.bytes[Bytes - 1U];
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

static_assert(sizeof(HandlerProbe<128U>) == 128U);
static_assert(sizeof(HandlerProbe<129U>) == 129U);
static_assert(ai_factory::workbench::simulation::ObservationHandlerFor<
    HandlerProbe<128U>, StateProvider
>);
static_assert(!ai_factory::workbench::simulation::ObservationHandlerFor<
    HandlerProbe<129U>, StateProvider
>);
static_assert(
    ai_factory::workbench::closed_form::kMaximumThreadPreparedRowBytes == 256U
);
static_assert(
    ai_factory::workbench::monte_carlo::kMaximumSharedPreparedRowBytes == 2048U
);
static_assert(
    ai_factory::workbench::longstaff_schwartz::
        kMaximumSharedPreparedRowBytes == 2048U
);
static_assert(
    ai_factory::workbench::sample::kMaximumSharedPreparedInputBytes == 2048U
);

int main() {
    using namespace ai_factory::workbench;
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA device unavailable; skipping size-budget test.\n";
        return 77;
    }

    const CudaKernelLaunchDiagnostics handler = inspect_cuda_kernel_launch(
        thread_payload_kernel<128U>, dim3(1U), dim3(128U), 0U
    );
    const CudaKernelLaunchDiagnostics closed_form =
        inspect_cuda_kernel_launch(
            thread_payload_kernel<256U>, dim3(1U), dim3(128U), 0U
        );
    const CudaKernelLaunchDiagnostics monte_carlo =
        inspect_cuda_kernel_launch(
            shared_payload_kernel<2048U>, dim3(1U), dim3(128U), 0U
        );
    const CudaKernelLaunchDiagnostics longstaff_schwartz =
        inspect_cuda_kernel_launch(
            shared_payload_kernel<2048U>, dim3(1U), dim3(256U), 0U
        );
    const CudaKernelLaunchDiagnostics sample = inspect_cuda_kernel_launch(
        shared_payload_kernel<2048U>, dim3(1U), dim3(512U), 0U
    );

    require(
        monte_carlo.static_shared_bytes_per_block >= 2048U
            && longstaff_schwartz.static_shared_bytes_per_block >= 2048U
            && sample.static_shared_bytes_per_block >= 2048U,
        "A shared near-limit probe did not reserve its complete budget."
    );
    const auto resource = [](const CudaKernelLaunchDiagnostics& diagnostic) {
        return nlohmann::ordered_json{
            {"registers_per_thread", diagnostic.registers_per_thread},
            {"local_bytes_per_thread", diagnostic.local_bytes_per_thread},
            {"static_shared_bytes_per_block",
                diagnostic.static_shared_bytes_per_block},
            {"active_blocks_per_multiprocessor",
                diagnostic.active_blocks_per_multiprocessor},
            {"theoretical_occupancy", diagnostic.theoretical_occupancy},
        };
    };
    const nlohmann::ordered_json report{
        {"schema", "ai_factory_policy_size_budget"},
        {"gpu", handler.device_name},
        {"compute_capability",
            std::to_string(handler.compute_capability_major) + "."
                + std::to_string(handler.compute_capability_minor)},
        {"budgets", {
            {"observation_handler_128", resource(handler)},
            {"closed_form_thread_row_256", resource(closed_form)},
            {"monte_carlo_shared_row_2048", resource(monte_carlo)},
            {"longstaff_schwartz_shared_row_2048",
                resource(longstaff_schwartz)},
            {"sample_shared_input_2048", resource(sample)},
        }},
    };
    std::cout << report.dump() << '\n';
    return 0;
}
