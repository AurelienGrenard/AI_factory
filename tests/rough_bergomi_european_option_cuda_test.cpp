// Exercise the unique fused-FFT pricing path for rough Bergomi and rough SABR.
#include "common/check_cuda.cuh"
#include "model/equity/rough/rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/rough_sabr/product/european_option.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

namespace bergomi =
    ai_factory::workbench::model::equity::rough_bergomi;
namespace sabr = ai_factory::workbench::model::equity::rough_sabr;

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
    check_cuda(availability, "rough Volterra pricing cudaGetDeviceCount");

    const bergomi::ModelParameters bergomi_model = {
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.10f, -0.70f,
    };
    sabr::ModelParameters sabr_model = {
        1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.10f, -0.70f, 1.0f,
    };
    product::EuropeanOptionParameters product = {1.0f, 126U};
    constexpr float day_fraction = 1.0f / 252.0f;
    constexpr float target_dt = 1.0f / 360.0f;
    constexpr std::size_t step_count = 180U;
    constexpr std::size_t path_count = 32'768U;
    constexpr std::size_t path_chunk_size = 8192U;
    constexpr std::uint64_t seed = 910000001ULL;
    const bergomi::WorkspacePlan workspace = bergomi::plan_pricing_workspace(
        2520U,
        path_count,
        path_chunk_size
    );
    require(
        workspace.convolution_bytes
            == ((path_chunk_size + 1U) / 2U) * 2520U * sizeof(float2),
        "rough workspace does not bound convolution storage by its chunk"
    );

    bergomi::ModelParameters* device_bergomi_model = nullptr;
    sabr::ModelParameters* device_sabr_model = nullptr;
    product::EuropeanOptionParameters* device_product = nullptr;
    void* device_workspace = nullptr;
    float* device_price = nullptr;
    float* device_error = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_bergomi_model, sizeof(bergomi_model)),
            "rough pricing Bergomi model malloc"
        );
        check_cuda(
            cudaMalloc(&device_sabr_model, sizeof(sabr_model)),
            "rough pricing SABR model malloc"
        );
        check_cuda(
            cudaMalloc(&device_product, sizeof(product)),
            "rough pricing product malloc"
        );
        check_cuda(
            cudaMalloc(&device_workspace, workspace.workspace_bytes),
            "rough pricing workspace malloc"
        );
        check_cuda(cudaMalloc(&device_price, sizeof(float)), "price malloc");
        check_cuda(cudaMalloc(&device_error, sizeof(float)), "error malloc");
        check_cuda(
            cudaMemcpy(
                device_bergomi_model,
                &bergomi_model,
                sizeof(bergomi_model),
                cudaMemcpyHostToDevice
            ),
            "rough pricing Bergomi model copy"
        );
        check_cuda(
            cudaMemcpy(
                device_sabr_model,
                &sabr_model,
                sizeof(sabr_model),
                cudaMemcpyHostToDevice
            ),
            "rough pricing SABR model copy"
        );
        check_cuda(
            cudaMemcpy(
                device_product,
                &product,
                sizeof(product),
                cudaMemcpyHostToDevice
            ),
            "rough pricing product copy"
        );

        auto read_result = [&] {
            std::pair<float, float> result{};
            check_cuda(
                cudaMemcpy(
                    &result.first,
                    device_price,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "rough pricing price copy"
            );
            check_cuda(
                cudaMemcpy(
                    &result.second,
                    device_error,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "rough pricing error copy"
            );
            return result;
        };
        auto launch_bergomi = [&](auto side_tag, std::size_t steps) {
            constexpr OptionSide side = decltype(side_tag)::value;
            bergomi::launch_rough_bergomi_european_option_cuda<side>(
                device_bergomi_model, 1U, device_product, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U,
                path_count, day_fraction, target_dt, steps,
                path_chunk_size, device_workspace,
                workspace.workspace_bytes, seed, device_price, device_error
            );
            check_cuda(cudaDeviceSynchronize(), "rough Bergomi synchronize");
            return read_result();
        };
        auto launch_sabr = [&](auto side_tag, std::size_t steps) {
            constexpr OptionSide side = decltype(side_tag)::value;
            sabr::launch_rough_sabr_european_option_cuda<side>(
                device_sabr_model, 1U, device_product, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U,
                path_count, day_fraction, target_dt, steps,
                path_chunk_size, device_workspace,
                workspace.workspace_bytes, seed, device_price, device_error
            );
            check_cuda(cudaDeviceSynchronize(), "rough SABR synchronize");
            return read_result();
        };
        const auto call =
            std::integral_constant<OptionSide, OptionSide::call>{};
        const auto put =
            std::integral_constant<OptionSide, OptionSide::put>{};

        const auto bergomi_call = launch_bergomi(call, step_count);
        const auto bergomi_replay = launch_bergomi(call, step_count);
        const auto sabr_call = launch_sabr(call, step_count);
        const auto bergomi_put = launch_bergomi(put, step_count);
        require(
            bergomi_call == bergomi_replay,
            "fused rough-Bergomi pricing does not replay bit for bit"
        );
        require(
            std::fabs(bergomi_call.first - sabr_call.first) < 2.0e-5f
                && std::fabs(bergomi_call.second - sabr_call.second)
                    < 2.0e-5f,
            "rough SABR beta=1 price does not reduce to rough Bergomi"
        );
        const float maturity = product.maturity_days * day_fraction;
        const float expected_parity =
            std::exp(-bergomi_model.dividend_yield * maturity)
            - product.strike
                * std::exp(-bergomi_model.risk_free_rate * maturity);
        require(
            std::fabs(
                (bergomi_call.first - bergomi_put.first) - expected_parity
            ) < 0.02f,
            "fused rough-Bergomi prices violate call-put parity"
        );

        sabr_model.beta = 0.70f;
        check_cuda(
            cudaMemcpy(
                device_sabr_model,
                &sabr_model,
                sizeof(sabr_model),
                cudaMemcpyHostToDevice
            ),
            "rough pricing CEV SABR model copy"
        );
        const auto cev_call = launch_sabr(call, step_count);
        require(
            std::isfinite(cev_call.first) && cev_call.first >= 0.0f
                && std::isfinite(cev_call.second) && cev_call.second > 0.0f,
            "rough SABR CEV price statistics are invalid"
        );

        // Published stress edge: low spot, beta=0.5 and eta=5.  There is no
        // closed-form rough-SABR reference, so enforce a refinement bound in
        // addition to the beta=1 exact reduction above.
        sabr_model = {
            0.05f, 0.0f, 0.0f, 0.20f, 5.0f, 0.10f, -0.70f, 0.50f,
        };
        product = {0.05f, 126U};
        check_cuda(
            cudaMemcpy(
                device_sabr_model,
                &sabr_model,
                sizeof(sabr_model),
                cudaMemcpyHostToDevice
            ),
            "rough pricing stress SABR model copy"
        );
        check_cuda(
            cudaMemcpy(
                device_product,
                &product,
                sizeof(product),
                cudaMemcpyHostToDevice
            ),
            "rough pricing stress product copy"
        );
        const auto stress_90 = launch_sabr(call, 90U);
        const auto stress_180 = launch_sabr(call, 180U);
        const auto stress_360 = launch_sabr(call, 360U);
        const float coarse_refinement = std::fabs(
            stress_90.first - stress_180.first
        );
        const float fine_refinement = std::fabs(
            stress_180.first - stress_360.first
        );
        const float refinement_statistical_bound = 5.0f * std::hypot(
            stress_180.second, stress_360.second
        );
        std::cerr
            << "rough SABR beta=0.5 stress: 90=" << stress_90.first
            << ", 180=" << stress_180.first
            << ", 360=" << stress_360.first
            << ", fine bound=" << refinement_statistical_bound
            << '\n';
        require(
            fine_refinement
                <= coarse_refinement + refinement_statistical_bound,
            "rough SABR stress refinement does not contract statistically"
        );
        require(
            fine_refinement
                <= 0.10f * sabr_model.spot + refinement_statistical_bound,
            "rough SABR beta=0.5 stress refinement exceeds its price bound"
        );

        // Compile and execute every tuned FFT dispatch boundary.
        for (const std::size_t steps : {
                 90U, 180U, 360U, 720U, 1440U, 1800U, 2520U
             }) {
            const auto value = launch_sabr(call, steps);
            require(
                std::isfinite(value.first) && std::isfinite(value.second),
                "a fused rough FFT dispatch produced invalid statistics"
            );
        }
    } catch (...) {
        if (device_bergomi_model != nullptr) cudaFree(device_bergomi_model);
        if (device_sabr_model != nullptr) cudaFree(device_sabr_model);
        if (device_product != nullptr) cudaFree(device_product);
        if (device_workspace != nullptr) cudaFree(device_workspace);
        if (device_price != nullptr) cudaFree(device_price);
        if (device_error != nullptr) cudaFree(device_error);
        throw;
    }
    check_cuda(cudaFree(device_bergomi_model), "free Bergomi model");
    check_cuda(cudaFree(device_sabr_model), "free SABR model");
    check_cuda(cudaFree(device_product), "free product");
    check_cuda(cudaFree(device_workspace), "free workspace");
    check_cuda(cudaFree(device_price), "free price");
    check_cuda(cudaFree(device_error), "free error");
    return 0;
}
