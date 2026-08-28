// Offline rough-Heston CUDA probe. This is validation code, not a src target.
#include "common/check_cuda.cuh"
#include "model/equity/rough/rough_heston/product/european_option.cuh"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

using ai_factory::workbench::OptionSide;
using ai_factory::workbench::check_cuda;
using ai_factory::workbench::model::equity::rough_heston::ModelParameters;
using ai_factory::workbench::model::equity::rough_heston::PreparedDynamics;
using ai_factory::workbench::product::EuropeanOptionParameters;

constexpr std::size_t factor_count = 7U;
constexpr float kernel_fit_dt = 1.0f / 360.0f;
constexpr unsigned int threads_per_block = 256U;

struct Estimate {
    float price;
    float standard_error;
    float milliseconds;
};

template<OptionSide Side>
Estimate launch(
    const ModelParameters* device_model,
    const PreparedDynamics<factor_count>* device_prepared,
    const EuropeanOptionParameters* device_product,
    std::size_t path_count,
    std::uint64_t seed,
    float simulation_dt,
    std::uint32_t steps_per_day,
    float* device_price,
    float* device_standard_error
) {
    using ai_factory::workbench::model::equity::rough_heston::
        launch_rough_heston_european_option_cuda;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "rough-Heston probe create start");
    check_cuda(cudaEventCreate(&stop), "rough-Heston probe create stop");
    check_cuda(cudaEventRecord(start), "rough-Heston probe record start");
    launch_rough_heston_european_option_cuda<Side, factor_count>(
        device_model,
        1U,
        device_prepared,
        1U,
        device_product,
        1U,
        false,
        1U,
        0U,
        1U,
        path_count,
        simulation_dt,
        steps_per_day,
        threads_per_block,
        1U,
        seed,
        device_price,
        device_standard_error
    );
    check_cuda(cudaEventRecord(stop), "rough-Heston probe record stop");
    check_cuda(cudaEventSynchronize(stop), "rough-Heston probe synchronize");
    Estimate result{};
    check_cuda(
        cudaEventElapsedTime(&result.milliseconds, start, stop),
        "rough-Heston probe elapsed time"
    );
    check_cuda(
        cudaMemcpy(
            &result.price,
            device_price,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "rough-Heston probe copy price"
    );
    check_cuda(
        cudaMemcpy(
            &result.standard_error,
            device_standard_error,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "rough-Heston probe copy standard error"
    );
    check_cuda(cudaEventDestroy(stop), "rough-Heston probe destroy stop");
    check_cuda(cudaEventDestroy(start), "rough-Heston probe destroy start");
    if (!std::isfinite(result.price) || result.price < 0.0f
        || !std::isfinite(result.standard_error)
        || result.standard_error <= 0.0f) {
        throw std::runtime_error("rough-Heston probe returned invalid moments");
    }
    return result;
}

}  // namespace

int main(int argument_count, char** arguments) {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::model::equity::rough_heston;
    const std::size_t path_count = argument_count > 1
        ? std::stoull(arguments[1])
        : 1U << 20U;
    const std::uint64_t seed = argument_count > 2
        ? std::stoull(arguments[2])
        : 912000001ULL;
    const std::uint32_t steps_per_day = argument_count > 3
        ? static_cast<std::uint32_t>(std::stoul(arguments[3]))
        : 1U;
    if (path_count < 2U) {
        throw std::invalid_argument("path count must be at least two");
    }
    if (steps_per_day < 1U) {
        throw std::invalid_argument("steps per day must be positive");
    }
    const float simulation_dt = 1.0f
        / (360.0f * static_cast<float>(steps_per_day));

    const ModelParameters model = {
        1.0f,
        0.02f,
        0.01f,
        0.04f,
        0.30f,
        0.02f,
        0.30f,
        0.10f,
        -0.70f,
    };
    const EuropeanOptionParameters product = {1.0f, 360U};
    const auto kernel = volterra::fit_positive_fractional_kernel_l2<
        factor_count
    >(model.hurst_exponent, 1.0f, kernel_fit_dt);
    const PreparedDynamics<factor_count> prepared = prepare_dynamics(
        model, kernel, 1.0f, simulation_dt
    );

    ModelParameters* device_model = nullptr;
    PreparedDynamics<factor_count>* device_prepared = nullptr;
    EuropeanOptionParameters* device_product = nullptr;
    float* device_price = nullptr;
    float* device_standard_error = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "probe model malloc");
    check_cuda(
        cudaMalloc(&device_prepared, sizeof(prepared)),
        "probe prepared malloc"
    );
    check_cuda(
        cudaMalloc(&device_product, sizeof(product)),
        "probe product malloc"
    );
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "probe price malloc");
    check_cuda(
        cudaMalloc(&device_standard_error, sizeof(float)),
        "probe standard-error malloc"
    );
    check_cuda(
        cudaMemcpy(
            device_model,
            &model,
            sizeof(model),
            cudaMemcpyHostToDevice
        ),
        "probe model copy"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            &prepared,
            sizeof(prepared),
            cudaMemcpyHostToDevice
        ),
        "probe prepared copy"
    );
    check_cuda(
        cudaMemcpy(
            device_product,
            &product,
            sizeof(product),
            cudaMemcpyHostToDevice
        ),
        "probe product copy"
    );

    const Estimate call = launch<OptionSide::call>(
        device_model,
        device_prepared,
        device_product,
        path_count,
        seed,
        simulation_dt,
        steps_per_day,
        device_price,
        device_standard_error
    );
    const Estimate put = launch<OptionSide::put>(
        device_model,
        device_prepared,
        device_product,
        path_count,
        seed,
        simulation_dt,
        steps_per_day,
        device_price,
        device_standard_error
    );

    check_cuda(cudaFree(device_standard_error), "probe standard-error free");
    check_cuda(cudaFree(device_price), "probe price free");
    check_cuda(cudaFree(device_product), "probe product free");
    check_cuda(cudaFree(device_prepared), "probe prepared free");
    check_cuda(cudaFree(device_model), "probe model free");

    const double maturity = 1.0;
    const double parity = model.spot * std::exp(-model.dividend_yield * maturity)
        - product.strike * std::exp(-model.risk_free_rate * maturity);
    std::printf(
        "{\n"
        "  \"factor_count\": %zu,\n"
        "  \"time_steps\": %u,\n"
        "  \"path_count\": %zu,\n"
        "  \"seed\": %llu,\n"
        "  \"call\": {\"price\": %.9g, \"standard_error\": %.9g, "
        "\"milliseconds\": %.9g},\n"
        "  \"put\": {\"price\": %.9g, \"standard_error\": %.9g, "
        "\"milliseconds\": %.9g},\n"
        "  \"call_put_parity_difference\": %.9g\n"
        "}\n",
        factor_count,
        360U * steps_per_day,
        path_count,
        static_cast<unsigned long long>(seed),
        call.price,
        call.standard_error,
        call.milliseconds,
        put.price,
        put.standard_error,
        put.milliseconds,
        static_cast<double>(call.price - put.price) - parity
    );
    return 0;
}
