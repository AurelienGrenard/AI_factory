// Compare the G2 caplet CUDA launcher with an FP64 formula.
#include "common/check_cuda.cuh"
#include "model/fixed_income/g2/rate_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kDayFraction = 1.0 / 252.0;

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Return B(delta) for the centered G2 short rate.
double integral_loading(double mean_reversion, double delta) {
    return -std::expm1(-mean_reversion * delta) / mean_reversion;
}

// Return one factor's future integral variance.
double integral_variance(double a, double sigma, double delta) {
    const double bracket =
        delta
        + 2.0 * std::expm1(-a * delta) / a
        - std::expm1(-2.0 * a * delta) / (2.0 * a);
    return sigma * sigma * bracket / (a * a);
}

// Return the variance of the integrated sum of both correlated factors.
double g2_integral_variance(
    const ai_factory::workbench::model::fixed_income::g2::ProcessParameters& process,
    double delta
) {
    const double a = process.mean_reversion_x;
    const double b = process.mean_reversion_y;
    const double cross = process.correlation
        * process.volatility_x * process.volatility_y / (a * b)
        * (
            delta
            - integral_loading(a, delta)
            - integral_loading(b, delta)
            + integral_loading(a + b, delta)
        );
    return integral_variance(a, process.volatility_x, delta)
        + integral_variance(b, process.volatility_y, delta)
        + 2.0 * cross;
}

// Price one G2 caplet through its equivalent zero-coupon put in FP64.
double caplet_price(
    const ai_factory::workbench::model::fixed_income::g2::
        ModelParameters& model,
    const ai_factory::workbench::product::RateOptionParameters& product
) {
    const double a = model.process.mean_reversion_x;
    const double b = model.process.mean_reversion_y;
    const double t1 = product.fixing_time * kDayFraction;
    const double t2 = product.payment_time * kDayFraction;
    const auto zero_coupon = [&](double maturity) {
        return std::exp(
            -integral_loading(a, maturity) * model.initial_state.state_x
            -integral_loading(b, maturity) * model.initial_state.state_y
            + 0.5 * g2_integral_variance(model.process, maturity)
        );
    };
    const double p01 = zero_coupon(t1);
    const double p02 = zero_coupon(t2);
    const double loading_x = integral_loading(a, t2 - t1);
    const double loading_y = integral_loading(b, t2 - t1);
    const double variance_x = model.process.volatility_x
        * model.process.volatility_x
        * (-std::expm1(-2.0 * a * t1)) / (2.0 * a);
    const double variance_y = model.process.volatility_y
        * model.process.volatility_y
        * (-std::expm1(-2.0 * b * t1)) / (2.0 * b);
    const double covariance = model.process.correlation
        * model.process.volatility_x * model.process.volatility_y
        * (-std::expm1(-(a + b) * t1)) / (a + b);
    const double volatility = std::sqrt(
        loading_x * loading_x * variance_x
        + loading_y * loading_y * variance_y
        + 2.0 * loading_x * loading_y * covariance
    );
    const double strike_factor =
        1.0 + product.accrual_period * kDayFraction * product.strike;
    const double bond_strike = 1.0 / strike_factor;
    if (volatility <= 1.0e-14) {
        const double put = std::max(bond_strike * p01 - p02, 0.0);
        return product.notional * strike_factor * put;
    }
    const double d1 =
        std::log(p02 / (bond_strike * p01)) / volatility
        + 0.5 * volatility;
    const double d2 = d1 - volatility;
    const auto normal_cdf = [](double value) {
        return 0.5 * std::erfc(-value / std::sqrt(2.0));
    };
    const double put =
        bond_strike * p01 * normal_cdf(-d2)
        - p02 * normal_cdf(-d1);
    return product.notional * strike_factor * put;
}

}  // namespace

// Verify aligned G2 caplet pricing against an independent CPU expression.
int main() {
    using namespace ai_factory::workbench;
    namespace g2 = model::fixed_income::g2;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "G2 caplet test cudaGetDeviceCount");

    const std::vector<ai_factory::workbench::model::fixed_income::g2::ModelParameters> models = {
        {{0.10f, 0.01f, 0.60f, 0.008f, -0.40f}, {0.02f, 0.01f}},
        {{0.25f, 0.015f, 0.90f, 0.010f, 0.20f}, {0.03f, 0.01f}},
        {{0.50f, 0.0f, 1.10f, 0.0f, 0.00f}, {0.02f, 0.005f}},
    };
    const std::vector<product::RateOptionParameters> products = {
        {1.0f, 0.0f, 126U, 252U, 126U},
        {1.0f, 0.04f, 252U, 378U, 126U},
        {1.0f, 0.06f, 504U, 567U, 63U},
    };
    constexpr std::size_t row_count = 3U;
    constexpr std::size_t cartesian_count = 6U;

    ai_factory::workbench::model::fixed_income::g2::ModelParameters* device_models = nullptr;
    product::RateOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_models, row_count * sizeof(models.front())),
            "G2 caplet test cudaMalloc models"
        );
        check_cuda(
            cudaMalloc(&device_products, row_count * sizeof(products.front())),
            "G2 caplet test cudaMalloc products"
        );
        check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "G2 caplet test cudaMalloc prices"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                row_count * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            "G2 caplet test cudaMemcpy models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                row_count * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            "G2 caplet test cudaMemcpy products"
        );

        ai_factory::workbench::model::fixed_income::g2::launch_g2_rate_option_cuda<OptionSide::call>(
            device_models,
            row_count,
            device_products,
            row_count,
            false,
            row_count,
            0U,
            row_count,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        std::vector<float> prices(row_count);
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                row_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "G2 caplet test cudaMemcpy prices"
        );

        for (std::size_t row = 0U; row < row_count; ++row) {
            const double expected = caplet_price(models[row], products[row]);
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f,
                "G2 caplet launcher returned an invalid price"
            );
            require(
                std::fabs(static_cast<double>(prices[row]) - expected)
                    < 5.0e-6,
                "G2 caplet CUDA price differs from the FP64 formula"
            );
        }

        // Exercise model-major Cartesian indexing across two launch batches.
        ai_factory::workbench::model::fixed_income::g2::launch_g2_rate_option_cuda<OptionSide::call>(
            device_models,
            2U,
            device_products,
            products.size(),
            true,
            cartesian_count,
            0U,
            2U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        ai_factory::workbench::model::fixed_income::g2::launch_g2_rate_option_cuda<OptionSide::call>(
            device_models,
            2U,
            device_products,
            products.size(),
            true,
            cartesian_count,
            2U,
            cartesian_count - 2U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        prices.resize(cartesian_count);
        check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                cartesian_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "G2 caplet test cudaMemcpy Cartesian prices"
        );
        for (std::size_t row = 0U; row < cartesian_count; ++row) {
            const std::size_t model_index = row / products.size();
            const std::size_t product_index = row % products.size();
            const double expected = caplet_price(
                models[model_index], products[product_index]
            );
            require(
                std::isfinite(prices[row])
                    && std::fabs(static_cast<double>(prices[row]) - expected)
                        < 5.0e-6,
                "G2 Cartesian caplet price differs from the FP64 formula"
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    check_cuda(cudaFree(device_models), "G2 caplet test cudaFree models");
    check_cuda(cudaFree(device_products), "G2 caplet test cudaFree products");
    check_cuda(cudaFree(device_prices), "G2 caplet test cudaFree prices");
}
