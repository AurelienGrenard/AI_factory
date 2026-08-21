// Compare three OU rate-option launchers with independent FP64 formulas.
#include "common/check_cuda.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/rate_option.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/zero_coupon_bond_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kDayFraction = 1.0 / 252.0;

using Model = ai_factory::workbench::model::ornstein_uhlenbeck::
    ModelParameters;

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

// Price a call (+1) or put (-1) on one zero-coupon bond in FP64.
double bond_option_price(
    const Model& model,
    double option_sign,
    double option_expiry,
    double bond_maturity,
    double strike
) {
    const double a = model.process.mean_reversion;
    const double sigma = model.process.volatility;
    const auto zero_coupon = [&](double maturity) {
        return std::exp(
            -integral_loading(a, maturity) * model.initial_state
            + 0.5 * integral_variance(a, sigma, maturity)
        );
    };
    const double expiry_bond = zero_coupon(option_expiry);
    const double underlying_bond = zero_coupon(bond_maturity);
    const double volatility = sigma
        * integral_loading(a, bond_maturity - option_expiry)
        * std::sqrt(-std::expm1(-2.0 * a * option_expiry) / (2.0 * a));
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
            "OU rate-option test cudaMalloc models"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_products, row_count * sizeof(Product)),
            "OU rate-option test cudaMalloc products"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "OU rate-option test cudaMalloc prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                row_count * sizeof(Model),
                cudaMemcpyHostToDevice
            ),
            "OU rate-option test cudaMemcpy models"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                row_count * sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "OU rate-option test cudaMemcpy products"
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
            "OU rate-option test cudaMemcpy aligned prices"
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
            "OU rate-option test cudaMemcpy Cartesian prices"
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
        cudaFree(device_models), "OU rate-option test cudaFree models"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_products), "OU rate-option test cudaFree products"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_prices), "OU rate-option test cudaFree prices"
    );
}

}  // namespace

// Validate floorlet and direct bond-option pricing against FP64 formulas.
int main() {
    using namespace ai_factory::workbench;
    namespace ou = model::ornstein_uhlenbeck;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "OU rate-option test cudaGetDeviceCount");

    const std::vector<Model> models = {
        {{0.10f, 0.01f}, 0.03f},
        {{0.25f, 0.015f}, 0.04f},
        {{0.50f, 0.0f}, 0.025f},
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
        ou::launch_ornstein_uhlenbeck_rate_option_cuda<OptionSide::put>,
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
        "OU floorlet CUDA price differs from the FP64 formula"
    );
    check_launcher(
        models,
        calls,
        ou::launch_ornstein_uhlenbeck_zero_coupon_bond_option_cuda<
            OptionSide::call
        >,
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
        "OU zero-coupon bond call differs from the FP64 formula"
    );
    check_launcher(
        models,
        puts,
        ou::launch_ornstein_uhlenbeck_zero_coupon_bond_option_cuda<
            OptionSide::put
        >,
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
        "OU zero-coupon bond put differs from the FP64 formula"
    );
}
