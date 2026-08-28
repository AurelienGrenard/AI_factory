// Compare OU Jamshidian swaption launchers with independent FP64 formulas.
#include "common/check_cuda.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/product/european_swaption.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using Model = ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck::
    ModelParameters;
using Product = ai_factory::workbench::product::
    RegularEuropeanSwaptionParameters;
using ExplicitProduct = ai_factory::workbench::product::
    ExplicitEuropeanSwaptionParameters;
using RegularLauncher = void (*)(
    const Model*, std::size_t, const Product*, std::size_t, ai_factory::workbench::PriceConstruction,
    std::size_t, std::size_t, std::size_t, float, unsigned int,
    std::size_t, float*
);
using ExplicitLauncher = void (*)(
    const Model*, std::size_t, const ExplicitProduct*,
    const std::uint32_t*, const float*, std::size_t, std::size_t, ai_factory::workbench::PriceConstruction,
    std::size_t, std::size_t, std::size_t, float, unsigned int,
    std::size_t, float*, std::uint32_t
);
constexpr double kDayFraction = 1.0 / 252.0;

struct ProductFixtures {
    std::vector<Product> products;
};

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
double jamshidian_boundary(
    const Model& model,
    const Product& product
) {
    const auto coupon_bond = [&](double state) {
        double value = 0.0;
        for (std::size_t payment = 0U;
             payment < product.payment_count;
            ++payment) {
            const double coefficient =
                product.strike * product.accrual_fraction
                + (payment + 1U == product.payment_count ? 1.0 : 0.0);
            const std::uint32_t payment_time_days = product.exercise_time_days
                + (payment + 1U) * product.payment_interval_days;
            value += coefficient * zero_coupon(
                model,
                state,
                product.exercise_time_days * kDayFraction,
                payment_time_days * kDayFraction
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
            product.strike * product.accrual_fraction
            + (payment + 1U == product.payment_count ? 1.0 : 0.0);
        if (coefficient == 0.0) continue;
        const std::uint32_t payment_time_days = product.exercise_time_days
            + (payment + 1U) * product.payment_interval_days;
        const double bond_strike = zero_coupon(
            model,
            boundary,
            product.exercise_time_days * kDayFraction,
            payment_time_days * kDayFraction
        );
        price += coefficient * bond_option_price(
            model,
            payer ? -1.0 : 1.0,
            product.exercise_time_days * kDayFraction,
            payment_time_days * kDayFraction,
            bond_strike
        );
    }
    return product.notional * price;
}

// Build one regular fixed-leg schedule.
void append_product(
    ProductFixtures& fixtures,
    float notional,
    float strike,
    std::uint32_t exercise_time_days,
    std::uint32_t payment_interval_days,
    std::uint32_t payment_count,
    float accrual_fraction
) {
    require(
        payment_interval_days > 0U
            && payment_count > 0U
            && accrual_fraction > 0.0f,
        "Invalid European swaption fixture schedule"
    );
    Product product{};
    product.notional = notional;
    product.strike = strike;
    product.accrual_fraction = accrual_fraction;
    product.exercise_time_days = exercise_time_days;
    product.payment_interval_days = payment_interval_days;
    product.payment_count = payment_count;
    fixtures.products.push_back(product);
}

// Exercise aligned and batched Cartesian indexing for one payoff side.
template <typename Launcher>
void check_launcher(
    const std::vector<Model>& models,
    const ProductFixtures& fixtures,
    Launcher launcher,
    bool payer,
    const char* mismatch_message
) {
    const std::vector<Product>& products = fixtures.products;
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
            ai_factory::workbench::PriceConstruction::CartesianProduct,
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
                models[model_index],
                products[product_index],
                payer
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

// Exercise the public explicit-schedule overload with a non-zero pool offset.
void check_explicit_launcher(
    const Model& model,
    const Product& regular_product,
    ExplicitLauncher launcher,
    bool payer
) {
    const std::size_t schedule_size =
        static_cast<std::size_t>(regular_product.payment_count) + 1U;
    std::vector<std::uint32_t> payment_times_days(schedule_size, 1U);
    std::vector<float> accrual_fractions(schedule_size, 1.0f);
    for (std::uint32_t payment = 0U;
         payment < regular_product.payment_count;
         ++payment) {
        payment_times_days[payment + 1U] = regular_product.exercise_time_days
            + (payment + 1U) * regular_product.payment_interval_days;
        accrual_fractions[payment + 1U] =
            regular_product.accrual_fraction;
    }
    const ExplicitProduct explicit_product = {
        regular_product.notional,
        regular_product.strike,
        regular_product.exercise_time_days,
        regular_product.payment_count,
        1U,
    };

    Model* device_model = nullptr;
    ExplicitProduct* device_product = nullptr;
    std::uint32_t* device_payment_times_days = nullptr;
    float* device_accrual_fractions = nullptr;
    float* device_price = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_model, sizeof(Model)),
            "Explicit OU swaption test cudaMalloc model"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_product, sizeof(ExplicitProduct)),
            "Explicit OU swaption test cudaMalloc product"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(
                &device_payment_times_days,
                schedule_size * sizeof(std::uint32_t)
            ),
            "Explicit OU swaption test cudaMalloc payment times"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(
                &device_accrual_fractions,
                schedule_size * sizeof(float)
            ),
            "Explicit OU swaption test cudaMalloc accrual fractions"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_price, sizeof(float)),
            "Explicit OU swaption test cudaMalloc price"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_model, &model, sizeof(Model), cudaMemcpyHostToDevice
            ),
            "Explicit OU swaption test cudaMemcpy model"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_product,
                &explicit_product,
                sizeof(ExplicitProduct),
                cudaMemcpyHostToDevice
            ),
            "Explicit OU swaption test cudaMemcpy product"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_payment_times_days,
                payment_times_days.data(),
                schedule_size * sizeof(std::uint32_t),
                cudaMemcpyHostToDevice
            ),
            "Explicit OU swaption test cudaMemcpy payment times"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_accrual_fractions,
                accrual_fractions.data(),
                schedule_size * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "Explicit OU swaption test cudaMemcpy accrual fractions"
        );
        launcher(
            device_model,
            1U,
            device_product,
            device_payment_times_days,
            device_accrual_fractions,
            schedule_size,
            1U,
            ai_factory::workbench::PriceConstruction::Aligned,
            1U,
            0U,
            1U,
            static_cast<float>(kDayFraction),
            32U,
            1U,
            device_price,
            regular_product.payment_count
        );
        float price = 0.0f;
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                &price,
                device_price,
                sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "Explicit OU swaption test cudaMemcpy price"
        );
        require(
            std::isfinite(price)
                && std::fabs(
                    static_cast<double>(price)
                    - swaption_price(model, regular_product, payer)
                ) < 5.0e-5,
            "Explicit OU swaption schedule differs from the regular schedule"
        );
    } catch (...) {
        if (device_model != nullptr) cudaFree(device_model);
        if (device_product != nullptr) cudaFree(device_product);
        if (device_payment_times_days != nullptr) {
            cudaFree(device_payment_times_days);
        }
        if (device_accrual_fractions != nullptr)
            cudaFree(device_accrual_fractions);
        if (device_price != nullptr) cudaFree(device_price);
        throw;
    }
    ai_factory::workbench::check_cuda(
        cudaFree(device_model), "Explicit OU swaption test cudaFree model"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_product), "Explicit OU swaption test cudaFree product"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_payment_times_days),
        "Explicit OU swaption test cudaFree payment times"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_accrual_fractions),
        "Explicit OU swaption test cudaFree accrual fractions"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_price), "Explicit OU swaption test cudaFree price"
    );
}

// Stress the shared Newton boundary with a 50-year monthly fixed leg.
void check_long_regular_schedule(
    const Model& model,
    RegularLauncher payer_launcher,
    RegularLauncher receiver_launcher
) {
    Product product{};
    product.notional = 1.0f;
    product.strike = 0.35f;
    product.accrual_fraction = 1.0f / 12.0f;
    product.exercise_time_days = 1260U;
    product.payment_interval_days = 21U;
    product.payment_count = 600U;

    Model* device_model = nullptr;
    Product* device_product = nullptr;
    float* device_price = nullptr;
    try {
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_model, sizeof(Model)),
            "Long OU swaption test cudaMalloc model"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_product, sizeof(Product)),
            "Long OU swaption test cudaMalloc product"
        );
        ai_factory::workbench::check_cuda(
            cudaMalloc(&device_price, sizeof(float)),
            "Long OU swaption test cudaMalloc price"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_model, &model, sizeof(Model), cudaMemcpyHostToDevice
            ),
            "Long OU swaption test cudaMemcpy model"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                device_product,
                &product,
                sizeof(Product),
                cudaMemcpyHostToDevice
            ),
            "Long OU swaption test cudaMemcpy product"
        );

        const auto launch = [&](RegularLauncher launcher) {
            launcher(
                device_model,
                1U,
                device_product,
                1U,
                ai_factory::workbench::PriceConstruction::Aligned,
                1U,
                0U,
                1U,
                static_cast<float>(kDayFraction),
                32U,
                1U,
                device_price
            );
            float price = 0.0f;
            ai_factory::workbench::check_cuda(
                cudaMemcpy(
                    &price,
                    device_price,
                    sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "Long OU swaption test cudaMemcpy price"
            );
            return price;
        };

        const float payer_price = launch(payer_launcher);
        const float receiver_price = launch(receiver_launcher);
        const double expected_payer = swaption_price(model, product, true);
        const double expected_receiver = swaption_price(
            model, product, false
        );
        require(
            std::isfinite(payer_price)
                && std::fabs(
                    static_cast<double>(payer_price) - expected_payer
                ) <= 5.0e-4 * std::max(1.0, std::fabs(expected_payer)),
            "Long OU payer swaption differs from the FP64 formula"
        );
        require(
            std::isfinite(receiver_price)
                && std::fabs(
                    static_cast<double>(receiver_price) - expected_receiver
                ) <= 5.0e-4 * std::max(1.0, std::fabs(expected_receiver)),
            "Long OU receiver swaption differs from the FP64 formula"
        );

        const double exercise_time =
            product.exercise_time_days * kDayFraction;
        double annuity = 0.0;
        for (std::uint32_t payment = 0U;
             payment < product.payment_count;
             ++payment) {
            const double payment_time = (
                product.exercise_time_days
                + (payment + 1U) * product.payment_interval_days
            ) * kDayFraction;
            annuity += product.accrual_fraction * zero_coupon(
                model, model.initial_state, 0.0, payment_time
            );
        }
        const double final_time = (
            product.exercise_time_days
            + product.payment_count * product.payment_interval_days
        ) * kDayFraction;
        const double payer_swap_value = zero_coupon(
            model, model.initial_state, 0.0, exercise_time
        ) - zero_coupon(
            model, model.initial_state, 0.0, final_time
        ) - product.strike * annuity;
        require(
            std::fabs(
                static_cast<double>(payer_price - receiver_price)
                - payer_swap_value
            ) <= 5.0e-4 * std::max(1.0, std::fabs(payer_swap_value)),
            "Long OU payer-receiver parity is violated"
        );
    } catch (...) {
        if (device_model != nullptr) cudaFree(device_model);
        if (device_product != nullptr) cudaFree(device_product);
        if (device_price != nullptr) cudaFree(device_price);
        throw;
    }
    ai_factory::workbench::check_cuda(
        cudaFree(device_model), "Long OU swaption test cudaFree model"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_product), "Long OU swaption test cudaFree product"
    );
    ai_factory::workbench::check_cuda(
        cudaFree(device_price), "Long OU swaption test cudaFree price"
    );
}

}  // namespace

// Validate payer and receiver Jamshidian prices in both construction modes.
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
    check_cuda(availability, "OU swaption test cudaGetDeviceCount");

    const std::vector<Model> models = {
        {{0.10f, 0.010f}, 0.030f},
        {{0.25f, 0.015f}, 0.040f},
        {{0.50f, 0.000f}, 0.025f},
    };
    ProductFixtures fixtures;
    append_product(
        fixtures,
        1.0f,
        0.025f,
        252U,
        126U,
        4U,
        0.50f
    );
    append_product(
        fixtures,
        2.0f,
        0.040f,
        504U,
        63U,
        8U,
        0.25f
    );
    append_product(
        fixtures,
        1.5f,
        0.060f,
        126U,
        126U,
        3U,
        0.50f
    );

    check_launcher(
        models,
        fixtures,
        static_cast<RegularLauncher>(
            ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
                SwaptionSide::payer
            >
        ),
        true,
        "OU payer swaption CUDA price differs from the FP64 formula"
    );
    check_launcher(
        models,
        fixtures,
        static_cast<RegularLauncher>(
            ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
                SwaptionSide::receiver
            >
        ),
        false,
        "OU receiver swaption CUDA price differs from the FP64 formula"
    );
    check_explicit_launcher(
        models.front(),
        fixtures.products.front(),
        static_cast<ExplicitLauncher>(
            ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
                SwaptionSide::payer
            >
        ),
        true
    );
    check_explicit_launcher(
        models.front(),
        fixtures.products.front(),
        static_cast<ExplicitLauncher>(
            ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
                SwaptionSide::receiver
            >
        ),
        false
    );
    check_long_regular_schedule(
        models.front(),
        static_cast<RegularLauncher>(
            ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
                SwaptionSide::payer
            >
        ),
        static_cast<RegularLauncher>(
            ou::launch_ornstein_uhlenbeck_european_swaption_cuda<
                SwaptionSide::receiver
            >
        )
    );
}
