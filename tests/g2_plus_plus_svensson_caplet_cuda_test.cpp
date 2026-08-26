// Compare the G2++ caplet CUDA launcher with one FP64 CPU formula.
#include "common/check_cuda.cuh"
#include "model/fixed_income/g2_plus_plus/svensson/rate_option.cuh"

#include <cuda_runtime.h>

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

// Evaluate one Svensson continuously compounded zero rate in FP64.
double zero_rate(
    const ai_factory::workbench::curve::svensson::SvenssonParameters& c,
    double maturity
) {
    const double x1 = maturity / static_cast<double>(c.tau1);
    const double x2 = maturity / static_cast<double>(c.tau2);
    const double loading1 = -std::expm1(-x1) / x1;
    const double loading2 = -std::expm1(-x2) / x2;
    return static_cast<double>(c.beta0)
        + static_cast<double>(c.beta1) * loading1
        + static_cast<double>(c.beta2) * (loading1 - std::exp(-x1))
        + static_cast<double>(c.beta3) * (loading2 - std::exp(-x2));
}

// Price one caplet through its equivalent zero-coupon put in FP64.
double caplet_price(
    const ai_factory::workbench::model::fixed_income::g2_plus_plus::ModelParameters& model,
    const ai_factory::workbench::curve::svensson::SvenssonParameters& curve,
    const ai_factory::workbench::product::RateOptionParameters& product
) {
    const double a = model.process.mean_reversion_x;
    const double b = model.process.mean_reversion_y;
    const double t1 = product.fixing_time * kDayFraction;
    const double t2 = product.payment_time * kDayFraction;
    const double p01 = std::exp(-t1 * zero_rate(curve, t1));
    const double p02 = std::exp(-t2 * zero_rate(curve, t2));
    const double loading_x = -std::expm1(-a * (t2 - t1)) / a;
    const double loading_y = -std::expm1(-b * (t2 - t1)) / b;
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

// Verify aligned pricing and the three-input construction count.
int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "caplet test cudaGetDeviceCount");

    const std::vector<model::fixed_income::g2_plus_plus::ModelParameters> models = {
        {{0.10f, 0.01f, 0.60f, 0.008f, -0.40f}},
        {{0.25f, 0.015f, 0.90f, 0.010f, 0.20f}},
        {{0.50f, 0.02f, 1.10f, 0.012f, -0.10f}},
    };
    const std::vector<curve::svensson::SvenssonParameters> curves = {
        {0.03f, -0.01f, 0.02f, -0.01f, 2.0f, 7.0f},
        {0.04f, -0.02f, 0.01f, 0.015f, 1.5f, 8.0f},
        {0.025f, 0.005f, -0.01f, 0.02f, 3.0f, 10.0f},
    };
    const std::vector<product::RateOptionParameters> products = {
        {1.0f, 0.02f, 126U, 252U, 126U},
        {1.0f, 0.04f, 252U, 378U, 126U},
        {1.0f, 0.06f, 504U, 567U, 63U},
    };
    constexpr std::size_t row_count = 3U;

    model::fixed_income::g2_plus_plus::ModelParameters* device_models = nullptr;
    curve::svensson::SvenssonParameters* device_curves = nullptr;
    product::RateOptionParameters* device_products = nullptr;
    float* device_prices = nullptr;
    try {
        check_cuda(
            cudaMalloc(&device_models, row_count * sizeof(models.front())),
            "caplet test cudaMalloc models"
        );
        check_cuda(
            cudaMalloc(&device_curves, row_count * sizeof(curves.front())),
            "caplet test cudaMalloc curves"
        );
        check_cuda(
            cudaMalloc(&device_products, row_count * sizeof(products.front())),
            "caplet test cudaMalloc products"
        );
        check_cuda(
            cudaMalloc(&device_prices, row_count * sizeof(float)),
            "caplet test cudaMalloc prices"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                row_count * sizeof(models.front()),
                cudaMemcpyHostToDevice
            ),
            "caplet test cudaMemcpy models"
        );
        check_cuda(
            cudaMemcpy(
                device_curves,
                curves.data(),
                row_count * sizeof(curves.front()),
                cudaMemcpyHostToDevice
            ),
            "caplet test cudaMemcpy curves"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                row_count * sizeof(products.front()),
                cudaMemcpyHostToDevice
            ),
            "caplet test cudaMemcpy products"
        );

        model::fixed_income::g2_plus_plus::svensson::
            launch_g2_plus_plus_svensson_rate_option_cuda<OptionSide::call>(
                device_models,
                row_count,
                device_curves,
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
            "caplet test cudaMemcpy prices"
        );

        for (std::size_t row = 0U; row < row_count; ++row) {
            const double expected = caplet_price(
                models[row], curves[row], products[row]
            );
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f,
                "Caplet launcher returned an invalid price"
            );
            require(
                std::fabs(static_cast<double>(prices[row]) - expected)
                    < 5.0e-6,
                "Caplet CUDA price differs from the FP64 formula"
            );
        }
    } catch (...) {
        if (device_models != nullptr) cudaFree(device_models);
        if (device_curves != nullptr) cudaFree(device_curves);
        if (device_products != nullptr) cudaFree(device_products);
        if (device_prices != nullptr) cudaFree(device_prices);
        throw;
    }

    check_cuda(cudaFree(device_models), "caplet test cudaFree models");
    check_cuda(cudaFree(device_curves), "caplet test cudaFree curves");
    check_cuda(cudaFree(device_products), "caplet test cudaFree products");
    check_cuda(cudaFree(device_prices), "caplet test cudaFree prices");
}
