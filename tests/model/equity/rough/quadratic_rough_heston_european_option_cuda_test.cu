// Qualify quadratic rough-Heston factor, time-step and H-grid convergence.
#include "common/check_cuda.cuh"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/quadratic_rough_heston/product/european_option.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace {

using namespace ai_factory::workbench;
namespace quadratic = model::equity::quadratic_rough_heston;

struct PriceResult {
    float price;
    float standard_error;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<OptionSide Side, std::size_t FactorCount>
PriceResult run_price(
    const quadratic::ModelParameters& model,
    const quadratic::PreparedDynamics<FactorCount>& prepared,
    const product::EuropeanOptionParameters& product,
    float dt,
    std::uint32_t simulation_steps_per_day
) {
    constexpr std::size_t path_count = 131'072U;
    constexpr unsigned int threads_per_block = 256U;
    constexpr std::size_t block_count = 1U;
    constexpr std::uint64_t seed = 11668828425518317568ULL;
    quadratic::ModelParameters* device_model = nullptr;
    quadratic::PreparedDynamics<FactorCount>* device_prepared = nullptr;
    product::EuropeanOptionParameters* device_product = nullptr;
    float* device_price = nullptr;
    float* device_standard_error = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "QRH price model");
    check_cuda(
        cudaMalloc(&device_prepared, sizeof(prepared)),
        "QRH price prepared dynamics"
    );
    check_cuda(
        cudaMalloc(&device_product, sizeof(product)),
        "QRH price product"
    );
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "QRH price output");
    check_cuda(
        cudaMalloc(&device_standard_error, sizeof(float)),
        "QRH price standard error"
    );
    check_cuda(
        cudaMemcpy(device_model, &model, sizeof(model), cudaMemcpyHostToDevice),
        "QRH price model copy"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            &prepared,
            sizeof(prepared),
            cudaMemcpyHostToDevice
        ),
        "QRH price prepared copy"
    );
    check_cuda(
        cudaMemcpy(
            device_product,
            &product,
            sizeof(product),
            cudaMemcpyHostToDevice
        ),
        "QRH price product copy"
    );
    quadratic::launch_quadratic_rough_heston_european_option_cuda<
        Side,
        FactorCount
    >(
        device_model,
        1U,
        device_prepared,
        1U,
        &product,
        device_product,
        1U,
        PriceConstruction::Aligned,
        1U,
        0U,
        1U,
        path_count,
        dt,
        simulation_steps_per_day,
        threads_per_block,
        block_count,
        seed,
        device_price,
        device_standard_error
    );
    check_cuda(cudaDeviceSynchronize(), "QRH price synchronize");
    PriceResult result{};
    check_cuda(
        cudaMemcpy(
            &result.price,
            device_price,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "QRH price copy"
    );
    check_cuda(
        cudaMemcpy(
            &result.standard_error,
            device_standard_error,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "QRH price standard-error copy"
    );
    require(
        std::isfinite(result.price) && result.price >= 0.0f
            && std::isfinite(result.standard_error)
            && result.standard_error > 0.0f,
        "Quadratic rough-Heston returned invalid price moments."
    );
    check_cuda(cudaFree(device_standard_error), "QRH price error free");
    check_cuda(cudaFree(device_price), "QRH price output free");
    check_cuda(cudaFree(device_product), "QRH price product free");
    check_cuda(cudaFree(device_prepared), "QRH price prepared free");
    check_cuda(cudaFree(device_model), "QRH price model free");
    return result;
}

double statistical_bound(const PriceResult& first, const PriceResult& second) {
    return 5.0 * std::hypot(
        static_cast<double>(first.standard_error),
        static_cast<double>(second.standard_error)
    );
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "quadratic rough-Heston price cudaGetDeviceCount");

    const quadratic::ModelParameters model{
        1.0f, 0.02f, 0.01f, 0.15f, 0.60f,
        0.08f, 0.005f, 1.50f, 1.00f, 0.105371f,
    };
    const product::EuropeanOptionParameters product{1.0f, 252U};
    constexpr float dt = 1.0f / 504.0f;
    const auto prepared_2 = quadratic::prepare_dynamics<2U>(model, 1.0f, dt);
    const auto prepared_3 = quadratic::prepare_dynamics<3U>(model, 1.0f, dt);
    const auto prepared_7 = quadratic::prepare_dynamics<7U>(model, 1.0f, dt);
    const PriceResult factors_2 = run_price<OptionSide::call>(
        model, prepared_2, product, dt, 2U
    );
    const PriceResult factors_3 = run_price<OptionSide::call>(
        model, prepared_3, product, dt, 2U
    );
    const PriceResult factors_7 = run_price<OptionSide::call>(
        model, prepared_7, product, dt, 2U
    );
    const double factor_coarse = std::abs(factors_3.price - factors_2.price);
    const double factor_fine = std::abs(factors_7.price - factors_3.price);
    require(
        factor_fine <= factor_coarse
            + statistical_bound(factors_2, factors_3)
            + statistical_bound(factors_3, factors_7),
        "Quadratic rough-Heston price did not converge with factor count."
    );

    constexpr float coarse_dt = 1.0f / 252.0f;
    constexpr float fine_dt = 1.0f / 1008.0f;
    const auto prepared_coarse = quadratic::prepare_dynamics<7U>(
        model, 1.0f, coarse_dt
    );
    const auto prepared_fine = quadratic::prepare_dynamics<7U>(
        model, 1.0f, fine_dt
    );
    const PriceResult time_coarse = run_price<OptionSide::call>(
        model, prepared_coarse, product, coarse_dt, 1U
    );
    const PriceResult time_middle = factors_7;
    const PriceResult time_fine = run_price<OptionSide::call>(
        model, prepared_fine, product, fine_dt, 4U
    );
    const double time_coarse_delta = std::abs(
        time_middle.price - time_coarse.price
    );
    const double time_fine_delta = std::abs(
        time_fine.price - time_middle.price
    );
    require(
        time_fine_delta <= time_coarse_delta
            + statistical_bound(time_coarse, time_middle)
            + statistical_bound(time_middle, time_fine),
        "Quadratic rough-Heston price did not converge with time refinement."
    );

    const auto grid =
        volterra::fit_positive_fractional_kernel_l2_hurst_grid<7U, 257U>(
            0.01f,
            0.20f,
            1.0f,
            dt
        );
    const auto prepared_grid = quadratic::prepare_dynamics(
        model,
        grid.interpolate(model.hurst_exponent),
        dt
    );
    const PriceResult grid_price = run_price<OptionSide::call>(
        model, prepared_grid, product, dt, 2U
    );
    require(
        std::abs(grid_price.price - factors_7.price)
            <= statistical_bound(grid_price, factors_7),
        "Quadratic rough-Heston H-grid changed the price distribution."
    );

    const PriceResult replay = run_price<OptionSide::call>(
        model, prepared_7, product, dt, 2U
    );
    require(
        replay.price == factors_7.price
            && replay.standard_error == factors_7.standard_error,
        "Quadratic rough-Heston price replay is not bitwise deterministic."
    );
    const PriceResult put = run_price<OptionSide::put>(
        model, prepared_7, product, dt, 2U
    );
    const float expected_parity = model.spot * std::exp(
        -model.dividend_yield
    ) - product.strike * std::exp(-model.risk_free_rate);
    require(
        std::abs((factors_7.price - put.price) - expected_parity)
            <= statistical_bound(factors_7, put),
        "Quadratic rough-Heston call-put parity failed."
    );

    std::printf(
        "QUADRATIC_ROUGH_HESTON_CONVERGENCE,N2=%.8g,N3=%.8g,N7=%.8g,"
        "dt252=%.8g,dt504=%.8g,dt1008=%.8g,grid=%.8g\n",
        factors_2.price,
        factors_3.price,
        factors_7.price,
        time_coarse.price,
        time_middle.price,
        time_fine.price,
        grid_price.price
    );
}
