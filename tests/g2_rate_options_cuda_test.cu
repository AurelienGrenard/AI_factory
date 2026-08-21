// Compare three G2 rate-option launchers with independent FP64 formulas.
#include "common/check_cuda.cuh"
#include "model/fixed_income/g2/rate_option.cuh"
#include "model/fixed_income/g2/zero_coupon_bond_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kDayFraction = 1.0 / 252.0;

using Model = ai_factory::workbench::model::g2::
    ModelParameters;

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
double g2_integral_variance(const Model& model, double delta) {
    const double a = model.process.mean_reversion_x;
    const double b = model.process.mean_reversion_y;
    const double cross = model.process.correlation
        * model.process.volatility_x * model.process.volatility_y / (a * b)
        * (
            delta
            - integral_loading(a, delta)
            - integral_loading(b, delta)
            + integral_loading(a + b, delta)
        );
    return integral_variance(a, model.process.volatility_x, delta)
        + integral_variance(b, model.process.volatility_y, delta)
        + 2.0 * cross;
}

// Price a call (+1) or put (-1) on one zero-coupon bond in FP64.
double bond_option_price(
    const Model& model,
    double option_sign,
    double option_expiry,
    double bond_maturity,
    double strike
) {
    const double a = model.process.mean_reversion_x;
    const double b = model.process.mean_reversion_y;
    const auto zero_coupon = [&](double maturity) {
        return std::exp(
            -integral_loading(a, maturity) * model.initial_state.state_x
            -integral_loading(b, maturity) * model.initial_state.state_y
            + 0.5 * g2_integral_variance(model, maturity)
        );
    };
    const double expiry_bond = zero_coupon(option_expiry);
    const double underlying_bond = zero_coupon(bond_maturity);
    const double loading_x = integral_loading(
        a, bond_maturity - option_expiry
    );
    const double loading_y = integral_loading(
        b, bond_maturity - option_expiry
    );
    const double variance_x = model.process.volatility_x
        * model.process.volatility_x
        * (-std::expm1(-2.0 * a * option_expiry)) / (2.0 * a);
    const double variance_y = model.process.volatility_y
        * model.process.volatility_y
        * (-std::expm1(-2.0 * b * option_expiry)) / (2.0 * b);
    const double covariance = model.process.correlation
        * model.process.volatility_x * model.process.volatility_y
        * (-std::expm1(-(a + b) * option_expiry)) / (a + b);
    const double volatility = std::sqrt(
        loading_x * loading_x * variance_x
        + loading_y * loading_y * variance_y
        + 2.0 * loading_x * loading_y * covariance
    );
    if (volatility <= 1.0e-14) {
        return std::max(
            option_sign * (underlying_bond - strike * expiry_bond), 0.0
        );
    }
    const double d1 =
        std::log(underlying_bond / (strike * expiry_bond)) / volatility
        + 0.5 * volatility;
    const double d2 = d1 - volatility;
    const auto normal_cdf = [](double value) {
        return 0.5 * std::erfc(-value / std::sqrt(2.0));
    };
    return option_sign
        * (
            underlying_bond * normal_cdf(option_sign * d1)
            - strike * expiry_bond * normal_cdf(option_sign * d2)
        );
}

// Exercise aligned and Cartesian indexing for one analytical launcher.
template <typename Product, typename Launcher, typename Expected>
void check_launcher(
    const std::vector<Model>& models,
    const std::vector<Product>& products,
    Launcher launcher,
    Expected expected,
    const char* mismatch_message
) {
    constexpr std::size_t row_count = 3U;
    constexpr std::size_t cartesian_count = 6U;
    Model* device_models = nullptr;
    Product* device_products = nullptr;
    float* device_prices = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_models, row_count * sizeof(Model)),
            "G2 rate-option test cudaMalloc models"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_products, row_count * sizeof(Product)),
            "G2 rate-option test cudaMalloc products"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "G2 rate-option test cudaMalloc prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                row_count * sizeof(Model),
                cudaMemcpyHostToDevice
            ),
            "G2 rate-option test cudaMemcpy models"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                row_count * sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "G2 rate-option test cudaMemcpy products"
        );

        launcher(
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
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                row_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "G2 rate-option test cudaMemcpy aligned prices"
        );
        for (std::size_t row = 0U; row < row_count; ++row) {
            require(
                std::isfinite(prices[row])
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(models[row], products[row])
                    ) < 5.0e-6,
                mismatch_message
            );
        }

        // Split the model-major Cartesian product across two launch batches.
        launcher(
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
        launcher(
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
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                cartesian_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "G2 rate-option test cudaMemcpy Cartesian prices"
        );
        for (std::size_t row = 0U; row < cartesian_count; ++row) {
            const std::size_t model_index = row / products.size();
            const std::size_t product_index = row % products.size();
            require(
                std::isfinite(prices[row])
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(
                            models[model_index], products[product_index]
                        )
                    ) < 5.0e-6,
                mismatch_message
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    ai_factory::workbench::check_cuda(
        cudaFree(device_models), "G2 rate-option test cudaFree models"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_products), "G2 rate-option test cudaFree products"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_prices), "G2 rate-option test cudaFree prices"
    );
}

}  // namespace

// Validate floorlet and direct bond-option pricing against FP64 formulas.
int main() {
    using namespace ai_factory::workbench;
    namespace g2 = model::g2;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "G2 rate-option test cudaGetDeviceCount");

    const std::vector<Model> models = {
        {{0.10f, 0.01f, 0.60f, 0.008f, -0.40f}, {0.02f, 0.01f}},
        {{0.25f, 0.015f, 0.90f, 0.010f, 0.20f}, {0.03f, 0.01f}},
        {{0.50f, 0.0f, 1.10f, 0.0f, 0.00f}, {0.02f, 0.005f}},
    };
    const std::vector<product::RateOptionParameters> floorlets = {
        {1.0f, 0.00f, 126U, 252U, 126U},
        {2.0f, 0.04f, 252U, 378U, 126U},
        {1.5f, 0.08f, 504U, 567U, 63U},
    };
    const std::vector<product::ZeroCouponBondOptionParameters> calls = {
        {1.0f, 0.90f, 126U, 252U},
        {2.0f, 0.97f, 252U, 378U},
        {1.5f, 1.05f, 504U, 756U},
    };
    const std::vector<product::ZeroCouponBondOptionParameters> puts = {
        {1.0f, 0.90f, 126U, 252U},
        {2.0f, 0.97f, 252U, 378U},
        {1.5f, 1.05f, 504U, 756U},
    };

    check_launcher(
        models,
        floorlets,
        g2::launch_g2_rate_option_cuda<OptionSide::put>,
        [](const Model& model, const product::RateOptionParameters& product) {
            const double strike_factor =
                1.0 + product.accrual_period * kDayFraction * product.strike;
            return product.notional * strike_factor * bond_option_price(
                model,
                1.0,
                product.fixing_time * kDayFraction,
                product.payment_time * kDayFraction,
                1.0 / strike_factor
            );
        },
        "G2 floorlet CUDA price differs from the FP64 formula"
    );
    check_launcher(
        models,
        calls,
        g2::launch_g2_zero_coupon_bond_option_cuda<OptionSide::call>,
        [](const Model& model,
           const product::ZeroCouponBondOptionParameters& product) {
            return product.notional * bond_option_price(
                model,
                1.0,
                product.option_expiry * kDayFraction,
                product.bond_maturity * kDayFraction,
                product.strike
            );
        },
        "G2 zero-coupon bond call differs from the FP64 formula"
    );
    check_launcher(
        models,
        puts,
        g2::launch_g2_zero_coupon_bond_option_cuda<OptionSide::put>,
        [](const Model& model,
           const product::ZeroCouponBondOptionParameters& product) {
            return product.notional * bond_option_price(
                model,
                -1.0,
                product.option_expiry * kDayFraction,
                product.bond_maturity * kDayFraction,
                product.strike
            );
        },
        "G2 zero-coupon bond put differs from the FP64 formula"
    );
}
