// Verify overflow-safe Volterra FFT workspace arithmetic before any CUDA allocation.
#include "common/volterra/hybrid_fft_workspace.cuh"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace {

namespace volterra = ai_factory::workbench::volterra;

__global__ void write_last_partial_moment(
    unsigned char* workspace,
    std::size_t partial_moment_offset,
    std::size_t partial_moment_count
) {
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        auto* partial_moments = reinterpret_cast<double2*>(
            workspace + partial_moment_offset
        );
        partial_moments[partial_moment_count - 1U] = make_double2(1.0, 2.0);
    }
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename Exception, typename Function>
void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Volterra workspace test cudaGetDeviceCount");

    constexpr std::size_t threads = volterra::kHybridFftPathThreads;
    require(
        volterra::hybrid_fft_partial_moment_count(threads) == 1U,
        "A full Volterra path block must emit one partial moment."
    );
    require(
        volterra::hybrid_fft_partial_moment_count(threads + 1U) == 2U,
        "A trailing Volterra path must emit a second partial moment."
    );

    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    require(
        volterra::hybrid_fft_partial_moment_count(maximum)
            == maximum / threads + 1U,
        "SIZE_MAX path ceiling division wrapped."
    );
    const std::size_t maximum_unsigned_blocks =
        std::numeric_limits<unsigned int>::max();
    const std::size_t maximum_unsigned_grid_paths =
        maximum_unsigned_blocks * threads;
    require(
        volterra::hybrid_fft_partial_moment_count(
            maximum_unsigned_grid_paths
        ) == maximum_unsigned_blocks,
        "Largest unsigned Volterra block count was computed incorrectly."
    );
    require_throws<std::overflow_error>(
        [maximum_unsigned_grid_paths] {
            volterra::validate_hybrid_fft_grid_x_size(
                volterra::hybrid_fft_partial_moment_count(
                    maximum_unsigned_grid_paths + 1U
                ),
                "test grid overflow"
            );
        },
        "Volterra block count beyond unsigned int reached device validation."
    );
    require_throws<std::overflow_error>(
        [maximum] {
            (void)volterra::hybrid_fft_convolution_bytes(1U, maximum);
        },
        "Oversized Volterra convolution storage was accepted."
    );
    require_throws<std::overflow_error>(
        [maximum] {
            (void)volterra::checked_hybrid_fft_sum(
                maximum,
                1U,
                "test overflow"
            );
        },
        "Overflowing Volterra workspace sum was accepted."
    );
    require_throws<std::overflow_error>(
        [maximum] {
            const std::size_t near_maximum_convolution_chunk =
                (maximum / sizeof(float2)) * 2U;
            (void)volterra::required_hybrid_fft_workspace_bytes(
                1U,
                2U,
                near_maximum_convolution_chunk
            );
        },
        "Overflowing Volterra convolution end offset was accepted."
    );
    require_throws<std::overflow_error>(
        [maximum] {
            (void)volterra::plan_hybrid_fft_workspace(1U, maximum, threads);
        },
        "SIZE_MAX Volterra path count reached allocation planning."
    );

    const std::size_t path_count = threads + 1U;
    const std::size_t expected = volterra::checked_hybrid_fft_sum(
        volterra::checked_hybrid_fft_sum(
            volterra::kHybridFftConvolutionOffset,
            volterra::hybrid_fft_convolution_bytes(3U, threads),
            "test convolution end"
        ),
        2U * 2U * sizeof(double),
        "test workspace end"
    );
    require(
        volterra::required_hybrid_fft_workspace_bytes(
            3U,
            path_count,
            threads
        ) == expected,
        "Trailing Volterra partial moment is absent from workspace bytes."
    );

    const volterra::HybridFftWorkspacePlan plan =
        volterra::plan_hybrid_fft_workspace(3U, path_count, threads);
    const std::size_t partial_moment_offset =
        volterra::checked_hybrid_fft_sum(
            volterra::kHybridFftConvolutionOffset,
            plan.convolution_bytes,
            "test partial-moment offset"
        );
    unsigned char* device_workspace = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_workspace, plan.workspace_bytes),
            "Volterra boundary workspace allocation"
        );
        write_last_partial_moment<<<1U, 1U>>>(
            device_workspace,
            partial_moment_offset,
            plan.partial_moment_count
        );
        check_cuda(
            cudaGetLastError(),
            "Volterra boundary workspace kernel launch"
        );
        check_cuda(
            cudaDeviceSynchronize(),
            "Volterra boundary workspace synchronization"
        );
        double2 last_partial_moment{};
        check_cuda(
            cudaMemcpy(
                &last_partial_moment,
                device_workspace + partial_moment_offset
                    + (plan.partial_moment_count - 1U) * sizeof(double2),
                sizeof(last_partial_moment),
                cudaMemcpyDeviceToHost
            ),
            "Volterra boundary partial-moment copy"
        );
        require(
            last_partial_moment.x == 1.0 && last_partial_moment.y == 2.0,
            "Last valid Volterra partial moment was not addressable."
        );
        check_cuda(
            cudaFree(device_workspace),
            "Volterra boundary workspace release"
        );
    } catch (...) {
        if (device_workspace != nullptr) {
            (void)cudaFree(device_workspace);
        }
        throw;
    }
}
