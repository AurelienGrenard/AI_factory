// Compare G2++ rate-option launchers with independent FP64 formulas.
#include "common/check_cuda.cuh"
#include "model/g2_plus_plus/nelson_siegel/floorlet.cuh"
#include "model/g2_plus_plus/nelson_siegel/zero_coupon_bond_call.cuh"
#include "model/g2_plus_plus/nelson_siegel/zero_coupon_bond_put.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using Model = ai_factory::workbench::model::g2_plus_plus::G2PlusPlusModelParameters;
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
    const double a = model.process.mean_reversion_x;
    const double b = model.process.mean_reversion_y;
    const double loading_x =
        -std::expm1(-a * (bond_maturity - option_expiry)) / a;
    const double loading_y =
        -std::expm1(-b * (bond_maturity - option_expiry)) / b;
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
            "G2++ rate-option test cudaMalloc models"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_curves, aligned_count * sizeof(Curve)),
            "G2++ rate-option test cudaMalloc curves"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_products, aligned_count * sizeof(Product)),
            "G2++ rate-option test cudaMalloc products"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "G2++ rate-option test cudaMalloc prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                aligned_count * sizeof(Model),
                cudaMemcpyHostToDevice
            ),
            "G2++ rate-option test cudaMemcpy models"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                aligned_count * sizeof(Curve),
                cudaMemcpyHostToDevice
            ),
            "G2++ rate-option test cudaMemcpy curves"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                aligned_count * sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "G2++ rate-option test cudaMemcpy products"
        );

        launcher(
            device_models,
            aligned_count,
            device_curves,
            aligned_count,
            device_products,
            aligned_count,
            false,
            aligned_count,
            0U,
            aligned_count,
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
            "G2++ rate-option test cudaMemcpy aligned prices"
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
            true,
            cartesian_count,
            0U,
            5U,
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
            true,
            cartesian_count,
            5U,
            cartesian_count - 5U,
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
            "G2++ rate-option test cudaMemcpy Cartesian prices"
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
        cudaFree(device_models), "G2++ rate-option test cudaFree models"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_curves), "G2++ rate-option test cudaFree curves"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_products),
        "G2++ rate-option test cudaFree products"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_prices), "G2++ rate-option test cudaFree prices"
    );
}

}  // namespace

// Validate floorlet and bond-option prices against FP64 formulas.
int main() {
    using namespace ai_factory::workbench;
    namespace g2pp = model::g2_plus_plus::nelson_siegel;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "G2++ rate-option test cudaGetDeviceCount");

    const std::vector<Model> models = {
        {{0.10f, 0.01f, 0.60f, 0.008f, -0.40f}},
        {{0.25f, 0.015f, 0.90f, 0.010f, 0.20f}},
        {{0.50f, 0.0f, 1.10f, 0.0f, 0.00f}},
    };
    const std::vector<Curve> curves = {
        {0.03f, -0.01f, 0.02f, 2.0f},
        {0.04f, -0.02f, 0.01f, 1.5f},
        {0.025f, 0.005f, -0.01f, 3.0f},
    };
    const std::vector<product::FloorletParameters> floorlets = {
        {1.0f, 0.00f, 0.5f, 1.0f, 0.5f},
        {2.0f, 0.04f, 1.0f, 1.5f, 0.5f},
        {1.5f, 0.08f, 2.0f, 2.25f, 0.25f},
    };
    const std::vector<product::ZeroCouponBondCallParameters> calls = {
        {1.0f, 0.90f, 0.5f, 1.0f},
        {2.0f, 0.97f, 1.0f, 1.5f},
        {1.5f, 1.05f, 2.0f, 3.0f},
    };
    const std::vector<product::ZeroCouponBondPutParameters> puts = {
        {1.0f, 0.90f, 0.5f, 1.0f},
        {2.0f, 0.97f, 1.0f, 1.5f},
        {1.5f, 1.05f, 2.0f, 3.0f},
    };

    check_launcher(
        models,
        curves,
        floorlets,
        g2pp::launch_g2_plus_plus_nelson_siegel_floorlet_cuda,
        [](const Model& model,
           const Curve& curve,
           const product::FloorletParameters& product) {
            const double strike_factor =
                1.0 + product.accrual_period * product.strike;
            return product.notional * strike_factor * bond_option_price(
                model,
                curve,
                1.0,
                product.fixing_time,
                product.payment_time,
                1.0 / strike_factor
            );
        },
        "G2++ floorlet CUDA price differs from the FP64 formula"
    );
    check_launcher(
        models,
        curves,
        calls,
        g2pp::launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_call_cuda,
        [](const Model& model,
           const Curve& curve,
           const product::ZeroCouponBondCallParameters& product) {
            return product.notional * bond_option_price(
                model,
                curve,
                1.0,
                product.option_expiry,
                product.bond_maturity,
                product.strike
            );
        },
        "G2++ zero-coupon bond call differs from the FP64 formula"
    );
    check_launcher(
        models,
        curves,
        puts,
        g2pp::launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_put_cuda,
        [](const Model& model,
           const Curve& curve,
           const product::ZeroCouponBondPutParameters& product) {
            return product.notional * bond_option_price(
                model,
                curve,
                -1.0,
                product.option_expiry,
                product.bond_maturity,
                product.strike
            );
        },
        "G2++ zero-coupon bond put differs from the FP64 formula"
    );
}
