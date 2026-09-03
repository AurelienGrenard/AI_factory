// Validate fixed-factor preparation and rough-Heston European pricing.
#include "common/check_cuda.cuh"
#include "common/volterra/fractional_kernel_approximation.hpp"
#include "model/equity/rough/rough_heston/european_option.cuh"
#include "model/equity/rough/rough_heston/numerics.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<std::size_t FactorCount>
void validate_preparation(
    const ai_factory::workbench::model::equity::rough_heston::ModelParameters&
        model,
    double& relative_error
) {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::model::equity::rough_heston;
    constexpr float dt = 1.0f / 360.0f;
    const auto kernel = volterra::fit_positive_fractional_kernel_l2<
        FactorCount
    >(model.hurst_exponent, 1.0f, dt);
    relative_error = volterra::fractional_kernel_relative_l2_error(
        kernel, model.hurst_exponent, 1.0f, dt
    );
    std::printf("ROUGH_HESTON_KERNEL,N=%zu", FactorCount);
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        std::printf(
            ",x%zu=%.8g,w%zu=%.8g",
            factor,
            kernel.nodes[factor],
            factor,
            kernel.weights[factor]
        );
    }
    std::printf("\n");
    float previous_node = 0.0f;
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        require(
            std::isfinite(kernel.nodes[factor])
                && kernel.nodes[factor] > previous_node
                && std::isfinite(kernel.weights[factor])
                && kernel.weights[factor] > 0.0f,
            "rough-Heston kernel rule is not positive and ordered"
        );
        previous_node = kernel.nodes[factor];
    }
    const PreparedDynamics<FactorCount> dynamics = prepare_dynamics(
        model, kernel, 1.0f, dt
    );
    double initial_variance = 0.0;
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        initial_variance += dynamics.kernel.weights[factor]
            * dynamics.initial_factors[factor];
        require(
            std::isfinite(dynamics.ode_half_shift[factor]),
            "rough-Heston ODE affine shift is not finite"
        );
        for (std::size_t column = 0U; column < FactorCount; ++column) {
            require(
                std::isfinite(
                    dynamics.ode_half_step[factor * FactorCount + column]
                ),
                "rough-Heston ODE matrix exponential is not finite"
            );
        }
    }
    require(
        std::abs(initial_variance - model.initial_variance) < 2.0e-6,
        "rough-Heston lifted initial variance is inconsistent"
    );
}

template<std::size_t FactorCount>
void exercise_gpu(
    const ai_factory::workbench::model::equity::rough_heston::ModelParameters&
        model,
    const ai_factory::workbench::product::EuropeanOptionParameters& product
) {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::model::equity::rough_heston;
    constexpr float dt = 1.0f / 360.0f;
    constexpr std::uint32_t steps_per_day = 1U;
    constexpr std::size_t path_count = 8'192U;
    constexpr unsigned int threads = 256U;
    constexpr std::uint64_t seed = 912000001ULL;
    const PreparedDynamics<FactorCount> prepared =
        prepare_dynamics<FactorCount>(model, 1.0f, dt);

    ModelParameters* device_model = nullptr;
    PreparedDynamics<FactorCount>* device_prepared = nullptr;
    product::EuropeanOptionParameters* device_product = nullptr;
    float* device_price = nullptr;
    float* device_error = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "rough-Heston model");
    check_cuda(
        cudaMalloc(&device_prepared, sizeof(prepared)),
        "rough-Heston prepared dynamics"
    );
    check_cuda(
        cudaMalloc(&device_product, sizeof(product)), "rough-Heston product"
    );
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "rough-Heston price");
    check_cuda(cudaMalloc(&device_error, sizeof(float)), "rough-Heston error");
    check_cuda(
        cudaMemcpy(device_model, &model, sizeof(model), cudaMemcpyHostToDevice),
        "rough-Heston copy model"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            &prepared,
            sizeof(prepared),
            cudaMemcpyHostToDevice
        ),
        "rough-Heston copy prepared dynamics"
    );
    check_cuda(
        cudaMemcpy(
            device_product,
            &product,
            sizeof(product),
            cudaMemcpyHostToDevice
        ),
        "rough-Heston copy product"
    );

    auto launch = [&](auto side_tag, float& milliseconds) {
        constexpr OptionSide side = decltype(side_tag)::value;
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check_cuda(cudaEventCreate(&start), "rough-Heston create start event");
        check_cuda(cudaEventCreate(&stop), "rough-Heston create stop event");
        check_cuda(cudaEventRecord(start), "rough-Heston record start event");
        launch_rough_heston_european_option_cuda<side, FactorCount>(
            device_model, 1U, device_prepared, 1U, device_product, 1U,
            ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U, 1U, path_count, dt,
            steps_per_day, threads, 1U,
            seed, device_price, device_error
        );
        check_cuda(cudaEventRecord(stop), "rough-Heston record stop event");
        check_cuda(
            cudaEventSynchronize(stop), "rough-Heston wait for stop event"
        );
        check_cuda(
            cudaEventElapsedTime(&milliseconds, start, stop),
            "rough-Heston elapsed time"
        );
        check_cuda(cudaEventDestroy(stop), "rough-Heston destroy stop event");
        check_cuda(cudaEventDestroy(start), "rough-Heston destroy start event");
        float price = 0.0f;
        float error = 0.0f;
        check_cuda(
            cudaMemcpy(
                &price, device_price, sizeof(float), cudaMemcpyDeviceToHost
            ),
            "rough-Heston copy price"
        );
        check_cuda(
            cudaMemcpy(
                &error, device_error, sizeof(float), cudaMemcpyDeviceToHost
            ),
            "rough-Heston copy error"
        );
        require(
            std::isfinite(price) && price >= 0.0f
                && std::isfinite(error) && error > 0.0f,
            "rough-Heston pricer returned invalid moments"
        );
        return price;
    };
    float call_milliseconds = 0.0f;
    float replay_milliseconds = 0.0f;
    float put_milliseconds = 0.0f;
    const float call = launch(
        std::integral_constant<OptionSide, OptionSide::call>{},
        call_milliseconds
    );
    const float replay = launch(
        std::integral_constant<OptionSide, OptionSide::call>{},
        replay_milliseconds
    );
    require(call == replay, "rough-Heston pricing is not bitwise reproducible");
    const float put = launch(
        std::integral_constant<OptionSide, OptionSide::put>{},
        put_milliseconds
    );
    const float maturity = static_cast<float>(product.maturity_days)
        * static_cast<float>(steps_per_day) * dt;
    const float expected_call_put_difference = model.spot * std::exp(
        -model.dividend_yield * maturity
    ) - product.strike * std::exp(-model.risk_free_rate * maturity);
    require(
        std::abs(
            (call - put) - expected_call_put_difference
        ) < 4.0e-3f,
        "rough-Heston call-put parity indicates an unstable lifted scheme"
    );
    std::printf(
        "ROUGH_HESTON_BENCH,N=%zu,steps=360,paths=%zu,call_ms=%.6f,"
        "replay_ms=%.6f,put_ms=%.6f,call=%.8f,put=%.8f\n",
        FactorCount,
        path_count,
        call_milliseconds,
        replay_milliseconds,
        put_milliseconds,
        call,
        put
    );

    check_cuda(cudaFree(device_error), "rough-Heston free error");
    check_cuda(cudaFree(device_price), "rough-Heston free price");
    check_cuda(cudaFree(device_product), "rough-Heston free product");
    check_cuda(cudaFree(device_prepared), "rough-Heston free prepared");
    check_cuda(cudaFree(device_model), "rough-Heston free model");
}

}  // namespace

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::model::equity::rough_heston;
    const ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 0.30f, 0.02f,
        0.30f, 0.10f, -0.70f,
    };
    double error_2 = 0.0;
    double error_3 = 0.0;
    double error_7 = 0.0;
    validate_preparation<2U>(model, error_2);
    validate_preparation<3U>(model, error_3);
    validate_preparation<7U>(model, error_7);
    require(
        error_7 < error_3 && error_3 < error_2 && error_7 < 5.0e-3,
        "rough-Heston factor refinement does not improve kernel L2 error"
    );
    std::printf(
        "ROUGH_HESTON_KERNEL_L2,N2=%.8f,N3=%.8f,N7=%.8f\n",
        error_2,
        error_3,
        error_7
    );

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "rough-Heston cudaGetDeviceCount");
    const product::EuropeanOptionParameters product = {1.0f, 360U};
    exercise_gpu<2U>(model, product);
    exercise_gpu<3U>(model, product);
    exercise_gpu<7U>(model, product);
    return 0;
}
