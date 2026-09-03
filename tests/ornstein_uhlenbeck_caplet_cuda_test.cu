// Compare the Ornstein-Uhlenbeck caplet CUDA launcher with an FP64 formula.
#include "common/check_cuda.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/rate_option.cuh"

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

// Return B(delta) for the centered Ornstein-Uhlenbeck short rate.
double integral_loading(double mean_reversion, double delta) {
    return -std::expm1(-mean_reversion * delta) / mean_reversion;
}

// Return the variance of the future integrated OU short rate.
double integral_variance(double a, double sigma, double delta) {
    const double bracket =
        delta
        + 2.0 * std::expm1(-a * delta) / a
        - std::expm1(-2.0 * a * delta) / (2.0 * a);
    return sigma * sigma * bracket / (a * a);
}

// Price one OU caplet through its equivalent zero-coupon put in FP64.
double caplet_price(
    const ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck::
        ModelParameters& model,
    const ai_factory::workbench::product::RateOptionParameters& product
) {
    const double a = model.process.mean_reversion;
    const double sigma = model.process.volatility;
    const double t1 = product.fixing_time_days * kDayFraction;
    const double t2 = product.payment_time_days * kDayFraction;
    const auto zero_coupon = [&](double maturity) {
        return std::exp(
            -integral_loading(a, maturity) * model.initial_state
            + 0.5 * integral_variance(a, sigma, maturity)
        );
    };
    const double p01 = zero_coupon(t1);
    const double p02 = zero_coupon(t2);
    const double volatility = sigma * integral_loading(a, t2 - t1)
        * std::sqrt(-std::expm1(-2.0 * a * t1) / (2.0 * a));
    const double strike_factor =
        1.0 + product.accrual_period_days * kDayFraction * product.strike;
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

// Verify aligned OU caplet pricing against an independent CPU expression.
int main() {
    using namespace ai_factory::workbench;
    namespace ou = model::fixed_income::ornstein_uhlenbeck;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "OU caplet test cudaGetDeviceCount");

    const std::vector<ou::ModelParameters> models = {
        {{0.10f, 0.01f}, 0.03f},
        {{0.25f, 0.015f}, 0.04f},
        {{0.50f, 0.0f}, 0.025f},
    };
    const std::vector<product::RateOptionParameters> products = {
        {1.0f, 0.0f, 126U, 252U, 126U},
        {1.0f, 0.04f, 252U, 378U, 126U},
        {1.0f, 0.06f, 504U, 567U, 63U},
    };
    constexpr std::size_t row_count = 3U;
    constexpr std::size_t cartesian_count = 6U;

    ou::ModelParameters* device_models = nullptr;
    product::RateOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_models, row_count * sizeof(models.front())),
            "OU caplet test cudaMalloc models"
        );
        check_cuda(
            cudaMalloc(&device_products, row_count * sizeof(products.front())),
            "OU caplet test cudaMalloc products"
        );
        check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "OU caplet test cudaMalloc prices"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                row_count * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            "OU caplet test cudaMemcpy models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                row_count * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            "OU caplet test cudaMemcpy products"
        );

        ou::launch_ornstein_uhlenbeck_rate_option_cuda<OptionSide::call>(
            device_models,
            row_count,
            device_products,
            row_count,
            ai_factory::workbench::PriceConstruction::Aligned,
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
            "OU caplet test cudaMemcpy prices"
        );

        for (std::size_t row = 0U; row < row_count; ++row) {
            const double expected = caplet_price(models[row], products[row]);
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f,
                "OU caplet launcher returned an invalid price"
            );
            require(
                std::fabs(static_cast<double>(prices[row]) - expected)
                    < 5.0e-6,
                "OU caplet CUDA price differs from the FP64 formula"
            );
        }

        // Exercise model-major Cartesian indexing across two launch batches.
        ou::launch_ornstein_uhlenbeck_rate_option_cuda<OptionSide::call>(
            device_models,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
            cartesian_count,
            0U,
            2U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        ou::launch_ornstein_uhlenbeck_rate_option_cuda<OptionSide::call>(
            device_models,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
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
            "OU caplet test cudaMemcpy Cartesian prices"
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
                "OU Cartesian caplet price differs from the FP64 formula"
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    check_cuda(cudaFree(device_models), "OU caplet test cudaFree models");
    check_cuda(cudaFree(device_products), "OU caplet test cudaFree products");
    check_cuda(cudaFree(device_prices), "OU caplet test cudaFree prices");
}
