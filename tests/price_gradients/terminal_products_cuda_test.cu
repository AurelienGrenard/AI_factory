// Generic terminal-product sensitivities over step and exact MC engines.
#include "model/equity/markovian/heston/product/asset_or_nothing_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/digital_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/digital_option_price_delta.cuh"
#include "model/equity/markovian/merton/product/digital_option_price_gradients.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <cmath>
#include <string_view>

using namespace price_gradient_test;
namespace heston = model::equity::heston;
namespace merton = model::equity::merton;

template<typename Plan>
void require_finite_diagonal(
    const Plan& plan,
    const DiagonalResults& result,
    const char* message
) {
    require(
        result.diagonal_hessian.size()
            == plan.result_count * plan.sensitivity_count(),
        "Unexpected diagonal-Hessian output size."
    );
    for (float value : result.diagonal_hessian) {
        require(std::isfinite(value), message);
    }
    for (float value : result.diagonal_hessian_error) {
        require(std::isfinite(value) && value >= 0.0f, message);
    }
}

template<OptionSide Side>
void check_heston_digital(std::size_t paths) {
    const std::vector<heston::ModelParameters> models{
        {1.05f, .03f, .01f, .04f, 1.5f, .04f, .3f, -.7f},
        {.9f, 0.f, 0.f, .09f, .8f, .06f, .5f, -.2f},
    };
    const std::vector<product::DigitalOptionParameters> products{
        {1.f, 16U, 2.f}, {.95f, 8U, .75f},
    };
    const pg::Sensitivity spot{"model.spot", {.005f}};
    const pg::PriceGradientConfiguration full{{
        spot,
        {"model.risk_free_rate", {.0001f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {.0001f, pg::BumpScale::absolute}},
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"model.kappa", {.005f}},
        {"model.theta", {.001f, pg::BumpScale::absolute}},
        {"model.gamma", {.005f}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"product.strike", {.005f}},
        {"product.cash_payoff", {.005f}},
        {"product.maturity_years", {
            1.f/504.f, pg::BumpScale::absolute
        }},
    }};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selected) {
        return heston::prepare_heston_digital_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selected
        );
    };
    const auto launcher =
        heston::launch_heston_digital_option_price_gradients_cuda<Side>;
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        128U, models.size(), 991U
    };

    const auto first = execute(prepare({{spot}}), launch, launcher);
    const auto price_only = execute(prepare({}), launch, launcher);
    DeviceArray<heston::ModelParameters> device_models(models);
    DeviceArray<product::DigitalOptionParameters> device_products(products);
    DeviceArray<float> legacy(4U * models.size());
    heston::launch_heston_digital_option_price_delta_cuda<Side>(
        models.data(), device_models.data, models.size(),
        products.data(), device_products.data, products.size(),
        PriceConstruction::Aligned, models.size(), 0U, models.size(), paths,
        1.f/504.f, 2U, 128U, models.size(), 991U, {.01f},
        legacy.data, legacy.data + models.size(),
        legacy.data + 2U*models.size(), legacy.data + 3U*models.size()
    );
    const auto reference = legacy.read();
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(price_only.price[row], first.price[row],
             "Digital price-only changed price");
        same(price_only.price_error[row], first.price_error[row],
             "Digital price-only changed price error");
        same(first.price[row], reference[row], "Digital legacy price");
        same(first.price_error[row], reference[models.size()+row],
             "Digital legacy price error");
        same(first.gradient[row], reference[2U*models.size()+row],
             "Digital legacy spot delta");
        same(first.gradient_error[row], reference[3U*models.size()+row],
             "Digital legacy spot-delta error");
    }

    const auto all = execute(prepare(full), launch, launcher);
    const auto diagonal_plan = heston::prepare_heston_digital_option_sensitivities(
        models, products, PriceConstruction::Aligned, {}, full,
        {pg::SensitivityOrders::first_and_second}
    );
    const auto combined = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan, launch,
        heston::launch_heston_digital_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto second_plan = heston::prepare_heston_digital_option_sensitivities(
        models, products, PriceConstruction::Aligned, {}, full,
        {pg::SensitivityOrders::second}
    );
    const auto second_only = execute_diagonal<pg::SensitivityOrders::second>(
        second_plan, launch,
        heston::launch_heston_digital_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::second
        >
    );
    require_finite_diagonal(
        diagonal_plan, combined, "Digital diagonal Hessian is invalid."
    );
    require(
        combined.price == second_only.price
            && combined.price_error == second_only.price_error
            && combined.diagonal_hessian == second_only.diagonal_hessian
            && combined.diagonal_hessian_error
                == second_only.diagonal_hessian_error,
        "Digital requested derivative orders changed shared results."
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(combined.price[row], all.price[row],
             "Digital diagonal changed central price");
        for (std::size_t sensitivity = 0U;
             sensitivity < full.sensitivities.size(); ++sensitivity) {
            const auto index = row*full.sensitivities.size()+sensitivity;
            same(combined.gradient[index], all.gradient[index],
                 "Digital diagonal changed first derivative");
            same(combined.gradient_error[index], all.gradient_error[index],
                 "Digital diagonal changed first-derivative error");
        }
        const auto cash_index = row*full.sensitivities.size()+9U;
        const float expected = combined.price[row] / products[row].cash_payoff;
        require(
            std::abs(combined.gradient[cash_index] - expected)
                <= 2.e-4f * std::max(1.0f, std::abs(expected)),
            "Digital cash-payoff derivative is inconsistent with linearity."
        );
    }
}

template<OptionSide Side>
void check_heston_asset_or_nothing(std::size_t paths) {
    const std::vector<heston::ModelParameters> models{
        {1.05f, .03f, .01f, .04f, 1.5f, .04f, .3f, -.7f},
        {.9f, 0.f, 0.f, .09f, .8f, .06f, .5f, -.2f},
    };
    const std::vector<product::AssetOrNothingOptionParameters> products{
        {1.f, 16U}, {.95f, 8U},
    };
    const pg::Sensitivity spot{"model.spot", {.005f}};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selected) {
        return heston::prepare_heston_asset_or_nothing_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selected
        );
    };
    const auto launcher =
        heston::launch_heston_asset_or_nothing_option_price_gradients_cuda<Side>;
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        128U, models.size(), 992U
    };
    const auto first = execute(prepare({{spot}}), launch, launcher);
    const auto price_only = execute(prepare({}), launch, launcher);
    DeviceArray<heston::ModelParameters> device_models(models);
    DeviceArray<product::AssetOrNothingOptionParameters> device_products(products);
    DeviceArray<float> legacy(4U * models.size());
    heston::launch_heston_asset_or_nothing_option_price_delta_cuda<Side>(
        models.data(), device_models.data, models.size(),
        products.data(), device_products.data, products.size(),
        PriceConstruction::Aligned, models.size(), 0U, models.size(), paths,
        1.f/504.f, 2U, 128U, models.size(), 992U, {.01f},
        legacy.data, legacy.data + models.size(),
        legacy.data + 2U*models.size(), legacy.data + 3U*models.size()
    );
    const auto reference = legacy.read();
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(price_only.price[row], first.price[row],
             "Asset-or-nothing price-only changed price");
        same(first.price[row], reference[row],
             "Asset-or-nothing legacy price");
        same(first.price_error[row], reference[models.size()+row],
             "Asset-or-nothing legacy price error");
        same(first.gradient[row], reference[2U*models.size()+row],
             "Asset-or-nothing legacy spot delta");
        same(first.gradient_error[row], reference[3U*models.size()+row],
             "Asset-or-nothing legacy spot-delta error");
    }
}

template<OptionSide Side>
void check_merton_digital(std::size_t paths) {
    const std::vector<merton::ModelParameters> models{
        {1.05f, .03f, .01f, .2f, .5f, -.1f, .25f},
        {.9f, 0.f, 0.f, .3f, 2.f, .02f, .4f},
    };
    const std::vector<product::DigitalOptionParameters> products{
        {1.f, 126U, 2.f}, {.95f, 252U, .75f},
    };
    const pg::Sensitivity spot{"model.spot", {.005f}};
    const pg::PriceGradientConfiguration full{{
        spot,
        {"model.jump_log_mean", {.002f, pg::BumpScale::absolute}},
        {"product.strike", {.005f}},
        {"product.cash_payoff", {.005f}},
    }};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selected) {
        return merton::prepare_merton_digital_option_price_gradients(
            models, products, PriceConstruction::Aligned, {}, selected
        );
    };
    const auto launcher =
        merton::launch_merton_digital_option_price_gradients_cuda<Side>;
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        256U, models.size(), 993U
    };
    const auto first = execute(prepare({{spot}}), launch, launcher);
    const auto all = execute(prepare(full), launch, launcher);
    DeviceArray<merton::ModelParameters> device_models(models);
    DeviceArray<product::DigitalOptionParameters> device_products(products);
    DeviceArray<float> legacy(4U * models.size());
    merton::launch_merton_digital_option_price_delta_cuda<Side>(
        models.data(), device_models.data, models.size(),
        products.data(), device_products.data, products.size(),
        PriceConstruction::Aligned, models.size(), 0U, models.size(), paths,
        1.f/252.f, 256U, models.size(), 993U, {.01f},
        legacy.data, legacy.data + models.size(),
        legacy.data + 2U*models.size(), legacy.data + 3U*models.size()
    );
    const auto reference = legacy.read();
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(first.price[row], reference[row], "Merton digital legacy price");
        same(first.price_error[row], reference[models.size()+row],
             "Merton digital legacy price error");
        same(first.gradient[row], reference[2U*models.size()+row],
             "Merton digital legacy spot delta");
        same(first.gradient_error[row], reference[3U*models.size()+row],
             "Merton digital legacy spot-delta error");
    }
    const auto diagonal_plan = merton::prepare_merton_digital_option_sensitivities(
        models, products, PriceConstruction::Aligned, {}, full,
        {pg::SensitivityOrders::first_and_second}
    );
    const auto combined = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        diagonal_plan, launch,
        merton::launch_merton_digital_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    require_finite_diagonal(
        diagonal_plan, combined,
        "Merton digital diagonal Hessian is invalid."
    );
    require(
        combined.price == all.price
            && combined.price_error == all.price_error
            && combined.gradient == all.gradient
            && combined.gradient_error == all.gradient_error,
        "Merton digital derivative order changed first-order results."
    );
}

int main(int argc, char** argv) {
    try {
        std::size_t paths = 4097U;
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            paths = 513U;
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: test [--sanitizer]");
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        check_heston_digital<OptionSide::call>(paths);
        check_heston_digital<OptionSide::put>(paths);
        check_heston_asset_or_nothing<OptionSide::call>(paths);
        check_heston_asset_or_nothing<OptionSide::put>(paths);
        check_merton_digital<OptionSide::call>(paths);
        check_merton_digital<OptionSide::put>(paths);
        std::cout << "Terminal-product gradients: step/exact engines, both sides, "
                     "legacy parity and diagonal Hessians passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
