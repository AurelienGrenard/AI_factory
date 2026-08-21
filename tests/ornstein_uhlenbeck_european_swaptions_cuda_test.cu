// Compare OU Jamshidian swaption launchers with independent FP64 formulas.
#include "common/check_cuda.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/european_swaption.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using Model = ai_factory::workbench::model::ornstein_uhlenbeck::
    ModelParameters;
using Product = ai_factory::workbench::product::EuropeanSwaptionParameters;
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

// Evaluate one conditional zero-coupon bond in FP64.
double zero_coupon(
    const Model& model,
    double state,
    double valuation_time,
    double maturity
) {
    const double delta = maturity - valuation_time;
    return std::exp(
        -integral_loading(model.process.mean_reversion, delta) * state
        + 0.5 * integral_variance(
            model.process.mean_reversion,
            model.process.volatility,
            delta
        )
    );
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
    const double expiry_bond = zero_coupon(
        model, model.initial_state, 0.0, option_expiry
    );
    const double underlying_bond = zero_coupon(
        model, model.initial_state, 0.0, bond_maturity
    );
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

// Solve the coupon-bond boundary independently in FP64.
double jamshidian_boundary(const Model& model, const Product& product) {
    const auto coupon_bond = [&](double state) {
        double value = 0.0;
        for (std::size_t payment = 0U;
             payment < product.payment_count;
             ++payment) {
            const double coefficient =
                product.strike * product.accrual_periods[payment]
                    * kDayFraction
                + (payment + 1U == product.payment_count ? 1.0 : 0.0);
            value += coefficient * zero_coupon(
                model,
                state,
                product.exercise_time * kDayFraction,
                product.payment_times[payment] * kDayFraction
            );
        }
        return value;
    };

    double lower = -0.25;
    double upper = 0.25;
    double width = 0.25;
    for (unsigned int expansion = 0U;
         expansion < 32U && coupon_bond(lower) < 1.0;
         ++expansion) {
        width *= 2.0;
        lower -= width;
    }
    width = 0.25;
    for (unsigned int expansion = 0U;
         expansion < 32U && coupon_bond(upper) > 1.0;
         ++expansion) {
        width *= 2.0;
        upper += width;
    }
    require(
        coupon_bond(lower) >= 1.0 && coupon_bond(upper) <= 1.0,
        "FP64 Jamshidian test failed to bracket the state boundary"
    );
    for (unsigned int iteration = 0U; iteration < 100U; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        if (coupon_bond(middle) > 1.0)
            lower = middle;
        else
            upper = middle;
    }
    return 0.5 * (lower + upper);
}

// Price payer or receiver through the independent FP64 decomposition.
double swaption_price(
    const Model& model,
    const Product& product,
    bool payer
) {
    const double boundary = jamshidian_boundary(model, product);
    double price = 0.0;
    for (std::size_t payment = 0U;
         payment < product.payment_count;
         ++payment) {
        const double coefficient =
            product.strike * product.accrual_periods[payment] * kDayFraction
            + (payment + 1U == product.payment_count ? 1.0 : 0.0);
        if (coefficient == 0.0) continue;
        const double bond_strike = zero_coupon(
            model,
            boundary,
            product.exercise_time * kDayFraction,
            product.payment_times[payment] * kDayFraction
        );
        price += coefficient * bond_option_price(
            model,
            payer ? -1.0 : 1.0,
            product.exercise_time * kDayFraction,
            product.payment_times[payment] * kDayFraction,
            bond_strike
        );
    }
    return product.notional * price;
}

// Build one arbitrary fixed-leg schedule without relying on a regular grid.
Product make_product(
    float notional,
    float strike,
    std::uint32_t exercise_time,
    const std::vector<std::uint32_t>& payment_times,
    const std::vector<std::uint32_t>& accrual_periods
) {
    require(
        !payment_times.empty()
            && payment_times.size() == accrual_periods.size()
            && payment_times.size()
                <= ai_factory::workbench::product::
                    kMaximumEuropeanSwaptionPayments,
        "Invalid European swaption fixture schedule"
    );
    Product product{};
    product.notional = notional;
    product.strike = strike;
    product.exercise_time = exercise_time;
    product.payment_count =
        static_cast<std::uint32_t>(payment_times.size());
    for (std::size_t payment = 0U;
         payment < payment_times.size();
         ++payment) {
        product.payment_times[payment] = payment_times[payment];
        product.accrual_periods[payment] = accrual_periods[payment];
    }
    return product;
}

// Exercise aligned and batched Cartesian indexing for one payoff side.
template <typename Launcher>
void check_launcher(
    const std::vector<Model>& models,
    const std::vector<Product>& products,
    Launcher launcher,
    bool payer,
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
            "OU swaption test cudaMalloc models"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_products, row_count * sizeof(Product)),
            "OU swaption test cudaMalloc products"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_prices, cartesian_count * sizeof(float)),
            "OU swaption test cudaMalloc prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                row_count * sizeof(Model),
                cudaMemcpyHostToDevice
            ),
            "OU swaption test cudaMemcpy models"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                row_count * sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "OU swaption test cudaMemcpy products"
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
            "OU swaption test cudaMemcpy aligned prices"
        );
        for (std::size_t row = 0U; row < row_count; ++row) {
            const double expected = swaption_price(
                models[row], products[row], payer
            );
            require(
                std::isfinite(prices[row]) && prices[row] >= 0.0f
                    && std::fabs(
                        static_cast<double>(prices[row]) - expected
                    ) < 5.0e-5,
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
            "OU swaption test cudaMemcpy Cartesian prices"
        );
        for (std::size_t row = 0U; row < cartesian_count; ++row) {
            const std::size_t model_index = row / products.size();
            const std::size_t product_index = row % products.size();
            const double expected = swaption_price(
                models[model_index], products[product_index], payer
            );
            require(
                std::isfinite(prices[row])
                    && std::fabs(
                        static_cast<double>(prices[row]) - expected
                    ) < 5.0e-5,
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
        cudaFree(device_models), "OU swaption test cudaFree models"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_products), "OU swaption test cudaFree products"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_prices), "OU swaption test cudaFree prices"
    );
}

}  // namespace

// Validate payer and receiver Jamshidian prices in both construction modes.
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
    check_cuda(availability, "OU swaption test cudaGetDeviceCount");

    const std::vector<Model> models = {
        {{0.10f, 0.010f}, 0.030f},
        {{0.25f, 0.015f}, 0.040f},
        {{0.50f, 0.000f}, 0.025f},
    };
    const std::vector<Product> products = {
        make_product(
            1.0f,
            0.025f,
            252U,
            {378U, 504U, 630U, 756U},
            {126U, 126U, 126U, 126U}
        ),
        make_product(
            2.0f,
            0.040f,
            504U,
            {567U, 693U, 819U, 1008U},
            {63U, 126U, 126U, 189U}
        ),
        make_product(
            1.5f,
            0.060f,
            126U,
            {252U, 378U, 504U},
            {126U, 126U, 126U}
        ),
    };

    check_launcher(
        models,
        products,
        ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
            OptionSide::call
        >,
        true,
        "OU payer swaption CUDA price differs from the FP64 formula"
    );
    check_launcher(
        models,
        products,
        ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
            OptionSide::put
        >,
        false,
        "OU receiver swaption CUDA price differs from the FP64 formula"
    );
}
