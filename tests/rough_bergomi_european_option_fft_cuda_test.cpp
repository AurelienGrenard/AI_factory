// Compare the direct-history and cuFFTDx rough-Bergomi European pricers.
#include "common/check_cuda.cuh"
#include "model/equity/rough_bergomi/european_option.cuh"
#include "model/equity/rough_bergomi/european_option_fft.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename Launch>
float measure_milliseconds(Launch&& launch) {
    using ai_factory::workbench::check_cuda;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "rough-Bergomi FFT test start event");
    check_cuda(cudaEventCreate(&stop), "rough-Bergomi FFT test stop event");
    check_cuda(cudaEventRecord(start), "rough-Bergomi FFT test record start");
    launch();
    check_cuda(cudaEventRecord(stop), "rough-Bergomi FFT test record stop");
    check_cuda(cudaEventSynchronize(stop), "rough-Bergomi FFT test wait");
    float milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start, stop),
        "rough-Bergomi FFT test elapsed time"
    );
    check_cuda(cudaEventDestroy(start), "rough-Bergomi FFT test destroy start");
    check_cuda(cudaEventDestroy(stop), "rough-Bergomi FFT test destroy stop");
    return milliseconds;
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::model::equity::rough_bergomi;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "rough-Bergomi FFT test cudaGetDeviceCount");

    const RoughBergomiModelParameters model = {
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.10f, -0.70f,
    };
    const product::EuropeanOptionParameters product = {1.0f, 252U};
    const product::EuropeanOptionParameters planning_product = {1.0f, 1764U};
    constexpr float day_fraction = 1.0f / 252.0f;
    constexpr float target_dt = 1.0f / 360.0f;
    constexpr std::size_t step_count = 360U;
    constexpr std::size_t path_count = 32'768U;
    constexpr unsigned int direct_threads = 128U;
    constexpr std::size_t direct_blocks = 1U;
    constexpr std::size_t fft_chunk_size = 32'768U;
    constexpr std::uint64_t seed = 910000001ULL;
    const RoughBergomiWorkspacePlan direct_plan =
        plan_european_option_workspace(
            &planning_product,
            1U,
            day_fraction,
            target_dt,
            direct_threads,
            direct_blocks
        );
    const RoughBergomiFftWorkspacePlan fft_plan =
        plan_european_option_fft_workspace(
            2520U, path_count, fft_chunk_size
        );

    RoughBergomiModelParameters* device_model = nullptr;
    product::EuropeanOptionParameters* device_product = nullptr;
    float* device_history = nullptr;
    void* device_fft_workspace = nullptr;
    float* device_direct_price = nullptr;
    float* device_direct_error = nullptr;
    float* device_fft_price = nullptr;
    float* device_fft_error = nullptr;
    try {
        check_cuda(cudaMalloc(&device_model, sizeof(model)), "FFT test model");
        check_cuda(
            cudaMalloc(&device_product, sizeof(product)), "FFT test product"
        );
        check_cuda(
            cudaMalloc(&device_history, direct_plan.history_bytes),
            "FFT test direct history"
        );
        check_cuda(
            cudaMalloc(&device_fft_workspace, fft_plan.workspace_bytes),
            "FFT test workspace"
        );
        check_cuda(
            cudaMalloc(&device_direct_price, sizeof(float)),
            "FFT test direct price"
        );
        check_cuda(
            cudaMalloc(&device_direct_error, sizeof(float)),
            "FFT test direct error"
        );
        check_cuda(
            cudaMalloc(&device_fft_price, sizeof(float)), "FFT test price"
        );
        check_cuda(
            cudaMalloc(&device_fft_error, sizeof(float)), "FFT test error"
        );
        check_cuda(
            cudaMemcpy(
                device_model, &model, sizeof(model), cudaMemcpyHostToDevice
            ),
            "FFT test copy model"
        );
        check_cuda(
            cudaMemcpy(
                device_product,
                &product,
                sizeof(product),
                cudaMemcpyHostToDevice
            ),
            "FFT test copy product"
        );

        auto direct_launch = [&](auto side_tag) {
            constexpr OptionSide side = decltype(side_tag)::value;
            launch_rough_bergomi_european_option_cuda<side>(
                device_model, 1U, device_product, 1U, false, 1U, 0U, 1U,
                path_count, day_fraction, target_dt, direct_threads, direct_blocks,
                direct_plan.maximum_step_count, device_history,
                direct_plan.history_float_count, seed, device_direct_price,
                device_direct_error
            );
        };
        auto fft_launch = [&](auto side_tag) {
            constexpr OptionSide side = decltype(side_tag)::value;
            launch_rough_bergomi_european_option_fft_cuda<side>(
                device_model, 1U, device_product, 1U, false, 1U, 0U,
                path_count, target_dt, step_count, fft_chunk_size,
                device_fft_workspace, fft_plan.workspace_bytes, seed,
                device_fft_price, device_fft_error
            );
        };
        const auto call =
            std::integral_constant<OptionSide, OptionSide::call>{};
        const auto put =
            std::integral_constant<OptionSide, OptionSide::put>{};

        direct_launch(call);
        fft_launch(call);
        check_cuda(cudaDeviceSynchronize(), "FFT test call warmup");
        const float direct_ms = measure_milliseconds([&] { direct_launch(call); });
        const float fft_ms = measure_milliseconds([&] { fft_launch(call); });

        auto read_pair = [&](float* price_pointer, float* error_pointer) {
            float price = 0.0f;
            float error = 0.0f;
            check_cuda(
                cudaMemcpy(
                    &price, price_pointer, sizeof(float), cudaMemcpyDeviceToHost
                ),
                "FFT test copy price"
            );
            check_cuda(
                cudaMemcpy(
                    &error, error_pointer, sizeof(float), cudaMemcpyDeviceToHost
                ),
                "FFT test copy error"
            );
            return std::pair<float, float>{price, error};
        };
        const auto direct_call = read_pair(
            device_direct_price, device_direct_error
        );
        const auto fft_call = read_pair(device_fft_price, device_fft_error);
        require(
            std::fabs(direct_call.first - fft_call.first) < 2.0e-4f
                && std::fabs(direct_call.second - fft_call.second) < 2.0e-4f,
            "cuFFTDx call does not match direct-history pricing"
        );

        fft_launch(call);
        check_cuda(cudaDeviceSynchronize(), "FFT test call replay");
        require(
            read_pair(device_fft_price, device_fft_error) == fft_call,
            "cuFFTDx call does not replay bit for bit"
        );

        direct_launch(put);
        fft_launch(put);
        check_cuda(cudaDeviceSynchronize(), "FFT test put synchronize");
        const auto direct_put = read_pair(
            device_direct_price, device_direct_error
        );
        const auto fft_put = read_pair(device_fft_price, device_fft_error);
        require(
            std::fabs(direct_put.first - fft_put.first) < 2.0e-4f
                && std::fabs(direct_put.second - fft_put.second) < 2.0e-4f,
            "cuFFTDx put does not match direct-history pricing"
        );
        require(
            std::isfinite(fft_call.first) && fft_call.first >= 0.0f
                && std::isfinite(fft_put.first) && fft_put.first >= 0.0f
                && fft_call.second > 0.0f && fft_put.second > 0.0f,
            "cuFFTDx call/put statistics are invalid"
        );

        // Exercise every production dispatch length against the original
        // convolution with the same paths and random stream.
        constexpr std::size_t dispatch_path_count = 512U;
        constexpr std::size_t dispatch_chunk_size = 512U;
        for (const std::size_t dispatch_steps : {
                 90U, 180U, 360U, 720U, 1440U, 1800U, 2520U
             }) {
            const product::EuropeanOptionParameters dispatch_product = {
                1.0f,
                static_cast<float>(dispatch_steps) / 360.0f,
            };
            check_cuda(
                cudaMemcpy(
                    device_product,
                    &dispatch_product,
                    sizeof(dispatch_product),
                    cudaMemcpyHostToDevice
                ),
                "FFT dispatch test copy product"
            );
            auto dispatch_compare = [&](auto side_tag) {
                constexpr OptionSide side = decltype(side_tag)::value;
                launch_rough_bergomi_european_option_cuda<side>(
                    device_model, 1U, device_product, 1U, false, 1U, 0U, 1U,
                    dispatch_path_count, day_fraction, target_dt, direct_threads,
                    direct_blocks, direct_plan.maximum_step_count,
                    device_history, direct_plan.history_float_count, seed,
                    device_direct_price, device_direct_error
                );
                check_cuda(
                    cudaDeviceSynchronize(), "FFT dispatch direct synchronize"
                );
                const auto direct_result = read_pair(
                    device_direct_price, device_direct_error
                );
                launch_rough_bergomi_european_option_fft_cuda<side>(
                    device_model, 1U, device_product, 1U, false, 1U, 0U,
                    dispatch_path_count, target_dt, dispatch_steps,
                    dispatch_chunk_size, device_fft_workspace,
                    fft_plan.workspace_bytes, seed, device_fft_price,
                    device_fft_error
                );
                check_cuda(
                    cudaDeviceSynchronize(), "FFT dispatch synchronize"
                );
                const auto fft_result = read_pair(
                    device_fft_price, device_fft_error
                );
                require(
                    std::fabs(direct_result.first - fft_result.first)
                            < 5.0e-4f
                        && std::fabs(
                            direct_result.second - fft_result.second
                        ) < 5.0e-4f,
                    "cuFFTDx dispatch does not match direct-history pricing"
                );
            };
            dispatch_compare(call);
            dispatch_compare(put);
        }
        std::printf(
            "ROUGH_BERGOMI_FFT_BENCH,N=%zu,paths=%zu,direct_ms=%.6f,"
            "fft_ms=%.6f,speedup=%.3f,call=%.8f,put=%.8f\n",
            step_count,
            path_count,
            direct_ms,
            fft_ms,
            direct_ms / fft_ms,
            fft_call.first,
            fft_put.first
        );
    } catch (...) {
        if (device_model != nullptr) cudaFree(device_model);
        if (device_product != nullptr) cudaFree(device_product);
        if (device_history != nullptr) cudaFree(device_history);
        if (device_fft_workspace != nullptr) cudaFree(device_fft_workspace);
        if (device_direct_price != nullptr) cudaFree(device_direct_price);
        if (device_direct_error != nullptr) cudaFree(device_direct_error);
        if (device_fft_price != nullptr) cudaFree(device_fft_price);
        if (device_fft_error != nullptr) cudaFree(device_fft_error);
        throw;
    }
    check_cuda(cudaFree(device_model), "FFT test free model");
    check_cuda(cudaFree(device_product), "FFT test free product");
    check_cuda(cudaFree(device_history), "FFT test free history");
    check_cuda(cudaFree(device_fft_workspace), "FFT test free workspace");
    check_cuda(cudaFree(device_direct_price), "FFT test free direct price");
    check_cuda(cudaFree(device_direct_error), "FFT test free direct error");
    check_cuda(cudaFree(device_fft_price), "FFT test free price");
    check_cuda(cudaFree(device_fft_error), "FFT test free error");
    return 0;
}
