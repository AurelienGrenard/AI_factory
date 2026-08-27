// Compare Hull-White rate-option launchers with independent FP64 formulas.
#include "common/check_cuda.cuh"
#include "model/fixed_income/hull_white/nelson_siegel/rate_option.cuh"
#include "model/fixed_income/hull_white/nelson_siegel/zero_coupon_bond_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kDayFraction = 1.0 / 252.0;

using Model = ai_factory::workbench::model::fixed_income::hull_white::ModelParameters;
using Curve = ai_factory::workbench::curve::nelson_siegel::
    NelsonSiegelParameters;

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Evaluate one Nelson-Siegel continuously compounded zero rate in FP64.
double zero_rate(const Curve& curve, double maturity) {
    const double x = maturity / static_cast<double>(curve.tau);
    const double loading = -std::expm1(-x) / x;
    return static_cast<double>(curve.beta0)
        + static_cast<double>(curve.beta1) * loading
        + static_cast<double>(curve.beta2) * (loading - std::exp(-x));
}

// Price a call (+1) or put (-1) on one fitted zero-coupon bond.
double bond_option_price(
    const Model& model,
    const Curve& curve,
    double option_sign,
    double option_expiry,
    double bond_maturity,
    double strike
) {
    const double expiry_bond = std::exp(
        -option_expiry * zero_rate(curve, option_expiry)
    );
    const double underlying_bond = std::exp(
        -bond_maturity * zero_rate(curve, bond_maturity)
    );
    const double a = model.mean_reversion;
    const double loading =
        -std::expm1(-a * (bond_maturity - option_expiry)) / a;
    const double volatility = model.volatility * loading
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

// Exercise aligned and three-input Cartesian indexing for one launcher.
template <typename Product, typename Launcher, typename Expected>
void check_launcher(
    const std::vector<Model>& models,
    const std::vector<Curve>& curves,
    const std::vector<Product>& products,
    Launcher launcher,
    Expected expected,
    const char* mismatch_message
) {
    constexpr std::size_t aligned_count = 3U;
    constexpr std::size_t cartesian_count = 12U;
    Model* device_models = nullptr;
    Curve* device_curves = nullptr;
    Product* device_products = nullptr;
    float* device_prices = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_models, aligned_count * sizeof(Model)),
            "Hull-White rate-option test cudaMalloc models"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_curves, aligned_count * sizeof(Curve)),
            "Hull-White rate-option test cudaMalloc curves"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_products, aligned_count * sizeof(Product)),
            "Hull-White rate-option test cudaMalloc products"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "Hull-White rate-option test cudaMalloc prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                aligned_count * sizeof(Model),
                cudaMemcpyHostToDevice
            ),
            "Hull-White rate-option test cudaMemcpy models"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                aligned_count * sizeof(Curve),
                cudaMemcpyHostToDevice
            ),
            "Hull-White rate-option test cudaMemcpy curves"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                aligned_count * sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "Hull-White rate-option test cudaMemcpy products"
        );

        launcher(
            device_models,
            aligned_count,
            device_curves,
            aligned_count,
            device_products,
            aligned_count,
            ai_factory::workbench::PriceConstruction::Aligned,
            aligned_count,
            0U,
            aligned_count,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        std::vector<float> prices(aligned_count);
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device_prices,
                aligned_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Hull-White rate-option test cudaMemcpy aligned prices"
        );
        for (std::size_t row = 0U; row < aligned_count; ++row) {
            require(
                std::isfinite(prices[row])
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(models[row], curves[row], products[row])
                    ) < 5.0e-6,
                mismatch_message
            );
        }

        // Split the model-major Cartesian product across two batches.
        launcher(
            device_models,
            2U,
            device_curves,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
            cartesian_count,
            0U,
            5U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_prices
        );
        launcher(
            device_models,
            2U,
            device_curves,
            2U,
            device_products,
            products.size(),
            ai_factory::workbench::PriceConstruction::CartesianProduct,
            cartesian_count,
            5U,
            cartesian_count - 5U,
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
            "Hull-White rate-option test cudaMemcpy Cartesian prices"
        );
        for (std::size_t row = 0U; row < cartesian_count; ++row) {
            const std::size_t curve_product_count =
                2U * products.size();
            const std::size_t model_index = row / curve_product_count;
            const std::size_t remainder = row % curve_product_count;
            const std::size_t curve_index = remainder / products.size();
            const std::size_t product_index = remainder % products.size();
            require(
                std::isfinite(prices[row])
                    && std::fabs(
                        static_cast<double>(prices[row])
                        - expected(
                            models[model_index],
                            curves[curve_index],
                            products[product_index]
                        )
                    ) < 5.0e-6,
                mismatch_message
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_curves != nullptr) cudaFree(device_curves);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    ai_factory::workbench::check_cuda(
        cudaFree(device_models), "Hull-White rate-option test cudaFree models"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_curves), "Hull-White rate-option test cudaFree curves"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_products),
        "Hull-White rate-option test cudaFree products"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_prices), "Hull-White rate-option test cudaFree prices"
    );
}

}  // namespace

// Validate floorlet and bond-option prices against FP64 formulas.
int main() {
    using namespace ai_factory::workbench;
    namespace hw = model::fixed_income::hull_white::nelson_siegel;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "Hull-White rate-option test cudaGetDeviceCount");

    const std::vector<Model> models = {
        {0.10f, 0.01f}, {0.25f, 0.015f}, {0.50f, 0.0f},
    };
    const std::vector<Curve> curves = {
        {0.03f, -0.01f, 0.02f, 2.0f},
        {0.04f, -0.02f, 0.01f, 1.5f},
        {0.025f, 0.005f, -0.01f, 3.0f},
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
        curves,
        floorlets,
        hw::launch_hull_white_nelson_siegel_rate_option_cuda<
            OptionSide::put
        >,
        [](const Model& model,
           const Curve& curve,
           const product::RateOptionParameters& product) {
            const double strike_factor =
                1.0 + product.accrual_period_days * kDayFraction * product.strike;
            return product.notional * strike_factor * bond_option_price(
                model,
                curve,
                1.0,
                product.fixing_time_days * kDayFraction,
                product.payment_time_days * kDayFraction,
                1.0 / strike_factor
            );
        },
        "Hull-White floorlet CUDA price differs from the FP64 formula"
    );
    check_launcher(
        models,
        curves,
        calls,
        hw::launch_hull_white_nelson_siegel_zero_coupon_bond_option_cuda<
            OptionSide::call
        >,
        [](const Model& model,
           const Curve& curve,
           const product::ZeroCouponBondOptionParameters& product) {
            return product.notional * bond_option_price(
                model,
                curve,
                1.0,
                product.option_expiry_days * kDayFraction,
                product.bond_maturity_days * kDayFraction,
                product.strike
            );
        },
        "Hull-White zero-coupon bond call differs from the FP64 formula"
    );
    check_launcher(
        models,
        curves,
        puts,
        hw::launch_hull_white_nelson_siegel_zero_coupon_bond_option_cuda<
            OptionSide::put
        >,
        [](const Model& model,
           const Curve& curve,
           const product::ZeroCouponBondOptionParameters& product) {
            return product.notional * bond_option_price(
                model,
                curve,
                -1.0,
                product.option_expiry_days * kDayFraction,
                product.bond_maturity_days * kDayFraction,
                product.strike
            );
        },
        "Hull-White zero-coupon bond put differs from the FP64 formula"
    );
}
