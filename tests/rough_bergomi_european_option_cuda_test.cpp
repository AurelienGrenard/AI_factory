// Exercise rough-Bergomi European call/put launch and workspace contracts.
#include "common/check_cuda.cuh"
#include "model/equity/rough_bergomi/european_option.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::rough_bergomi;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(
        availability,
        "rough-Bergomi European-option test cudaGetDeviceCount"
    );

    const RoughBergomiModelParameters model = {
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.10f, -0.70f,
    };
    const product::EuropeanOptionParameters product = {1.0f, 63U};
    constexpr float day_fraction = 1.0f / 252.0f;
    constexpr float target_dt = 1.0f / 64.0f;
    constexpr unsigned int threads_per_block = 128U;
    constexpr std::size_t block_count = 1U;
    constexpr std::size_t path_count = 4'096U;
    constexpr std::uint64_t seed = 910000001ULL;
    const RoughBergomiWorkspacePlan workspace =
        plan_european_option_workspace(
            &product,
            1U,
            day_fraction,
            target_dt,
            threads_per_block,
            block_count
        );
    require(
        workspace.maximum_step_count == 16U,
        "rough-Bergomi workspace planned an incorrect step count"
    );

    RoughBergomiModelParameters* device_model = nullptr;
    product::EuropeanOptionParameters* device_product = nullptr;
    float* device_history = nullptr;
    float* device_price = nullptr;
    float* device_standard_error = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_model, sizeof(model)),
            "rough-Bergomi European-option test model cudaMalloc"
        );
        check_cuda(
            cudaMalloc(&device_product, sizeof(product)),
            "rough-Bergomi European-option test product cudaMalloc"
        );
        check_cuda(
            cudaMalloc(&device_history, workspace.history_bytes),
            "rough-Bergomi European-option test history cudaMalloc"
        );
        check_cuda(
            cudaMalloc(&device_price, sizeof(float)),
            "rough-Bergomi European-option test price cudaMalloc"
        );
        check_cuda(
            cudaMalloc(&device_standard_error, sizeof(float)),
            "rough-Bergomi European-option test error cudaMalloc"
        );
        check_cuda(
            cudaMemcpy(
                device_model,
                &model,
                sizeof(model),
                cudaMemcpyHostToDevice
            ),
            "rough-Bergomi European-option test model cudaMemcpy"
        );
        check_cuda(
            cudaMemcpy(
                device_product,
                &product,
                sizeof(product),
                cudaMemcpyHostToDevice
            ),
            "rough-Bergomi European-option test product cudaMemcpy"
        );

        auto launch = [&](auto side_tag) {
            constexpr OptionSide side = decltype(side_tag)::value;
            launch_rough_bergomi_european_option_cuda<side>(
                device_model,
                1U,
                device_product,
                1U,
                false,
                1U,
                0U,
                1U,
                path_count,
                day_fraction,
                target_dt,
                threads_per_block,
                block_count,
                workspace.maximum_step_count,
                device_history,
                workspace.history_float_count,
                seed,
                device_price,
                device_standard_error
            );
            check_cuda(
                cudaDeviceSynchronize(),
                "rough-Bergomi European-option test synchronize"
            );
            float price = 0.0f;
            float standard_error = 0.0f;
            check_cuda(
                cudaMemcpy(
                    &price,
                    device_price,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "rough-Bergomi European-option test price cudaMemcpy"
            );
            check_cuda(
                cudaMemcpy(
                    &standard_error,
                    device_standard_error,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "rough-Bergomi European-option test error cudaMemcpy"
            );
            return std::pair<float, float>{price, standard_error};
        };

        const auto call_tag =
            std::integral_constant<OptionSide, OptionSide::call>{};
        const auto put_tag =
            std::integral_constant<OptionSide, OptionSide::put>{};
        const auto call_first = launch(call_tag);
        const auto call_replay = launch(call_tag);
        const auto put = launch(put_tag);
        require(
            call_first == call_replay,
            "rough-Bergomi European call does not replay bit for bit"
        );
        require(
            std::isfinite(call_first.first)
                && call_first.first >= 0.0f
                && std::isfinite(call_first.second)
                && call_first.second > 0.0f
                && std::isfinite(put.first)
                && put.first >= 0.0f
                && std::isfinite(put.second)
                && put.second > 0.0f,
            "rough-Bergomi European option statistics are invalid"
        );
        const float expected_parity =
            std::exp(
                -model.dividend_yield * product.maturity * day_fraction
            )
            - product.strike
                * std::exp(
                    -model.risk_free_rate * product.maturity * day_fraction
                );
        require(
            std::fabs((call_first.first - put.first) - expected_parity)
                < 0.02f,
            "rough-Bergomi European call/put prices violate parity"
        );
    } catch (...) {
        if (device_model != nullptr) cudaFree(device_model);
        if (device_product != nullptr) cudaFree(device_product);
        if (device_history != nullptr) cudaFree(device_history);
        if (device_price != nullptr) cudaFree(device_price);
        if (device_standard_error != nullptr) cudaFree(device_standard_error);
        throw;
    }
    check_cuda(cudaFree(device_model), "rough-Bergomi test model cudaFree");
    check_cuda(cudaFree(device_product), "rough-Bergomi test product cudaFree");
    check_cuda(cudaFree(device_history), "rough-Bergomi test history cudaFree");
    check_cuda(cudaFree(device_price), "rough-Bergomi test price cudaFree");
    check_cuda(
        cudaFree(device_standard_error),
        "rough-Bergomi test standard error cudaFree"
    );
    return 0;
}
