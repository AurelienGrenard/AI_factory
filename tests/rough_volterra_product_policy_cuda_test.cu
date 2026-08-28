// Prove that a dense barrier product plugs into the shared rough FFT engine.
#include "common/check_cuda.cuh"
#include "common/volterra/hybrid_schedule.cuh"
#include "model/equity/rough/rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/rough_bergomi/volterra_fft_pricing.cuh"
#include "product/asian_option/pricing_policy.cuh"
#include "product/forward_start_option/pricing_policy.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

namespace bergomi =
    ai_factory::workbench::model::equity::rough_bergomi;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "rough product-policy cudaGetDeviceCount");

    const bergomi::ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 1.0f, 0.10f, -0.60f,
    };
    const product::EuropeanOptionParameters european = {1.0f, 252U};
    product::UpAndOutOptionParameters barrier = {1.0f, 100.0f, 252U};
    const product::AsianOptionParameters asian = {1.0f, 252U};
    const product::ForwardStartOptionParameters forward_start = {
        1.0f, 126U, 252U,
    };
    constexpr std::size_t path_count = 4096U;
    constexpr std::size_t step_count = 32U;
    constexpr std::size_t path_chunk_size = 4096U;
    constexpr std::uint64_t seed = 912300001ULL;
    constexpr volterra::HybridTimeConfiguration time_configuration = {
        1.0f / 252.0f,
        1.0f / 32.0f,
    };
    const bergomi::WorkspacePlan workspace = bergomi::plan_pricing_workspace(
        step_count,
        path_count,
        path_chunk_size
    );

    bergomi::ModelParameters* device_model = nullptr;
    product::EuropeanOptionParameters* device_european = nullptr;
    product::UpAndOutOptionParameters* device_barrier = nullptr;
    product::AsianOptionParameters* device_asian = nullptr;
    product::ForwardStartOptionParameters* device_forward_start = nullptr;
    void* device_workspace = nullptr;
    float* device_price = nullptr;
    float* device_error = nullptr;
    try {
        check_cuda(cudaMalloc(&device_model, sizeof(model)), "model malloc");
        check_cuda(
            cudaMalloc(&device_european, sizeof(european)),
            "European product malloc"
        );
        check_cuda(
            cudaMalloc(&device_barrier, sizeof(barrier)),
            "barrier product malloc"
        );
        check_cuda(cudaMalloc(&device_asian, sizeof(asian)), "Asian malloc");
        check_cuda(
            cudaMalloc(&device_forward_start, sizeof(forward_start)),
            "forward-start malloc"
        );
        check_cuda(
            cudaMalloc(&device_workspace, workspace.workspace_bytes),
            "rough product workspace malloc"
        );
        check_cuda(cudaMalloc(&device_price, sizeof(float)), "price malloc");
        check_cuda(cudaMalloc(&device_error, sizeof(float)), "error malloc");
        check_cuda(
            cudaMemcpy(
                device_model,
                &model,
                sizeof(model),
                cudaMemcpyHostToDevice
            ),
            "model copy"
        );
        check_cuda(
            cudaMemcpy(
                device_european,
                &european,
                sizeof(european),
                cudaMemcpyHostToDevice
            ),
            "European product copy"
        );
        check_cuda(
            cudaMemcpy(
                device_asian,
                &asian,
                sizeof(asian),
                cudaMemcpyHostToDevice
            ),
            "Asian product copy"
        );
        check_cuda(
            cudaMemcpy(
                device_forward_start,
                &forward_start,
                sizeof(forward_start),
                cudaMemcpyHostToDevice
            ),
            "forward-start product copy"
        );

        auto read = [&] {
            std::pair<float, float> result{};
            check_cuda(
                cudaMemcpy(
                    &result.first,
                    device_price,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "price copy"
            );
            check_cuda(
                cudaMemcpy(
                    &result.second,
                    device_error,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "error copy"
            );
            return result;
        };

        bergomi::launch_rough_bergomi_european_option_cuda<OptionSide::call>(
            device_model, 1U, device_european, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U,
            path_count, time_configuration.day_fraction,
            time_configuration.target_dt, step_count, path_chunk_size,
            device_workspace, workspace.workspace_bytes, seed,
            device_price, device_error
        );
        check_cuda(cudaDeviceSynchronize(), "European synchronize");
        const auto vanilla = read();

        using BarrierPolicy = product::UpAndOutOptionPathPolicy<
            OptionSide::call
        >;
        auto launch_barrier = [&] {
            bergomi::launch_rough_bergomi_hybrid_fft_cuda<
                BarrierPolicy,
                volterra::DenseHybridSchedule
            >(
                device_model, 1U, device_barrier, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U,
                path_count, time_configuration, step_count,
                path_chunk_size, device_workspace,
                workspace.workspace_bytes, seed, device_price, device_error,
                "rough_bergomi.up_and_out_option", "call"
            );
            check_cuda(cudaDeviceSynchronize(), "barrier synchronize");
            return read();
        };
        check_cuda(
            cudaMemcpy(
                device_barrier,
                &barrier,
                sizeof(barrier),
                cudaMemcpyHostToDevice
            ),
            "high barrier copy"
        );
        const auto inactive_barrier = launch_barrier();
        require(
            inactive_barrier == vanilla,
            "an unreachable dense barrier changed the terminal payoff"
        );

        barrier.barrier = 0.5f;
        check_cuda(
            cudaMemcpy(
                device_barrier,
                &barrier,
                sizeof(barrier),
                cudaMemcpyHostToDevice
            ),
            "initially breached barrier copy"
        );
        const auto knocked_out = launch_barrier();
        require(
            knocked_out.first == 0.0f && knocked_out.second == 0.0f,
            "the dense barrier handler did not stop an initially dead path"
        );

        bergomi::launch_rough_bergomi_hybrid_fft_cuda<
            product::AsianOptionPathPolicy<OptionSide::call>,
            volterra::DenseHybridSchedule
        >(
            device_model, 1U, device_asian, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U,
            path_count, time_configuration, step_count, path_chunk_size,
            device_workspace, workspace.workspace_bytes, seed,
            device_price, device_error, "rough_bergomi.asian_option", "call"
        );
        check_cuda(cudaDeviceSynchronize(), "Asian synchronize");
        const auto asian_result = read();
        require(
            std::isfinite(asian_result.first)
                && std::isfinite(asian_result.second),
            "the dense Asian product produced invalid statistics"
        );

        bergomi::launch_rough_bergomi_hybrid_fft_cuda<
            product::ForwardStartOptionPathPolicy<OptionSide::call>,
            volterra::CalendarHybridSchedule<2U>
        >(
            device_model, 1U, device_forward_start, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U,
            path_count, time_configuration, step_count, path_chunk_size,
            device_workspace, workspace.workspace_bytes, seed,
            device_price, device_error,
            "rough_bergomi.forward_start_option", "call"
        );
        check_cuda(cudaDeviceSynchronize(), "forward-start synchronize");
        const auto forward_result = read();
        require(
            std::isfinite(forward_result.first)
                && std::isfinite(forward_result.second),
            "the two-date product produced invalid statistics"
        );
    } catch (...) {
        if (device_model != nullptr) cudaFree(device_model);
        if (device_european != nullptr) cudaFree(device_european);
        if (device_barrier != nullptr) cudaFree(device_barrier);
        if (device_asian != nullptr) cudaFree(device_asian);
        if (device_forward_start != nullptr) cudaFree(device_forward_start);
        if (device_workspace != nullptr) cudaFree(device_workspace);
        if (device_price != nullptr) cudaFree(device_price);
        if (device_error != nullptr) cudaFree(device_error);
        throw;
    }
    check_cuda(cudaFree(device_model), "free model");
    check_cuda(cudaFree(device_european), "free European product");
    check_cuda(cudaFree(device_barrier), "free barrier product");
    check_cuda(cudaFree(device_asian), "free Asian product");
    check_cuda(cudaFree(device_forward_start), "free forward-start product");
    check_cuda(cudaFree(device_workspace), "free workspace");
    check_cuda(cudaFree(device_price), "free price");
    check_cuda(cudaFree(device_error), "free error");
    return 0;
}
