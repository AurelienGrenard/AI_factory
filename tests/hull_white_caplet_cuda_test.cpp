// Compare the Hull-White caplet CUDA launcher with one FP64 CPU formula.
#include "common/check_cuda.cuh"
#include "model/hull_white/nelson_siegel/caplet.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Evaluate one Nelson-Siegel continuously compounded zero rate in FP64.
double zero_rate(
    const ai_factory::workbench::curve::nelson_siegel::NelsonSiegelParameters& c,
    double maturity
) {
    const double x = maturity / static_cast<double>(c.tau);
    const double loading = -std::expm1(-x) / x;
    return static_cast<double>(c.beta0)
        + static_cast<double>(c.beta1) * loading
        + static_cast<double>(c.beta2) * (loading - std::exp(-x));
}

// Price one caplet through its equivalent zero-coupon put in FP64.
double caplet_price(
    const ai_factory::workbench::model::hull_white::HullWhiteModelParameters& model,
    const ai_factory::workbench::curve::nelson_siegel::NelsonSiegelParameters& curve,
    const ai_factory::workbench::product::CapletParameters& product
) {
    const double a = model.mean_reversion;
    const double sigma = model.volatility;
    const double t1 = product.fixing_time;
    const double t2 = product.payment_time;
    const double p01 = std::exp(-t1 * zero_rate(curve, t1));
    const double p02 = std::exp(-t2 * zero_rate(curve, t2));
    const double loading = -std::expm1(-a * (t2 - t1)) / a;
    const double volatility = sigma * loading
        * std::sqrt(-std::expm1(-2.0 * a * t1) / (2.0 * a));
    const double strike_factor =
        1.0 + product.accrual_period * product.strike;
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

    const std::vector<model::hull_white::HullWhiteModelParameters> models = {
        {0.10f, 0.01f}, {0.25f, 0.015f}, {0.50f, 0.02f},
    };
    const std::vector<curve::nelson_siegel::NelsonSiegelParameters> curves = {
        {0.03f, -0.01f, 0.02f, 2.0f},
        {0.04f, -0.02f, 0.01f, 1.5f},
        {0.025f, 0.005f, -0.01f, 3.0f},
    };
    const std::vector<product::CapletParameters> products = {
        {1.0f, 0.02f, 0.5f, 1.0f, 0.5f},
        {1.0f, 0.04f, 1.0f, 1.5f, 0.5f},
        {1.0f, 0.06f, 2.0f, 2.25f, 0.25f},
    };
    constexpr std::size_t row_count = 3U;

    model::hull_white::HullWhiteModelParameters* device_models = nullptr;
    curve::nelson_siegel::NelsonSiegelParameters* device_curves = nullptr;
    product::CapletParameters* device_products = nullptr;
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

        model::hull_white::nelson_siegel::
            launch_hull_white_nelson_siegel_caplet_cuda(
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
