// SABR terminal sensitivities reuse the canonical transition and Philox path.
#include "model/equity/markovian/sabr/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/product/european_option.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

using namespace price_gradient_test;
namespace sabr = ai_factory::workbench::model::equity::sabr;

template<OptionSide Side>
void check(std::size_t paths) {
    const std::vector<sabr::ModelParameters> models{
        {1.0f, 0.0f, 0.0f, .25f, .4f, -.3f, .7f},
        {1.2f, .02f, .01f, .3f, 0.0f, 1.0f, 1.0f},
    };
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 8U}, {1.1f, 6U},
    };
    const pg::PriceGradientConfiguration selected{{
        {"model.spot", {.005}},
        {"model.risk_free_rate", {.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {.0005, pg::BumpScale::absolute}},
        {"model.initial_volatility", {.005}},
        {"model.volatility_of_volatility", {.005, pg::BumpScale::absolute}},
        {"model.rho", {.002, pg::BumpScale::absolute}},
        {"model.beta", {.002, pg::BumpScale::absolute}},
        {"product.strike", {.005}},
    }};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selection,
                             pg::SensitivityOrders orders) {
        return sabr::prepare_sabr_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, selection,
            {orders}
        );
    };
    const auto combined_plan = prepare(
        selected, pg::SensitivityOrders::first_and_second
    );
    const auto spot_task = sensitivity_task<
        pg::SensitivityOrders::first_and_second
    >(combined_plan, 0U, 0U);
    require(!spot_task.nodes[1U].reuse_central,
            "SABR spot bump incorrectly reuses a scaled path.");
    const auto rho_task = sensitivity_task<
        pg::SensitivityOrders::first_and_second
    >(combined_plan, 1U, 5U);
    const auto beta_task = sensitivity_task<
        pg::SensitivityOrders::first_and_second
    >(combined_plan, 1U, 6U);
    require(rho_task.stencil.kind == pg::StencilKind::backward
            && beta_task.stencil.kind == pg::StencilKind::backward,
            "SABR boundary stencils were not one-sided.");

    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths,
        128U, models.size(), 1709U, 1U,
    };
    const auto combined = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        combined_plan, launch,
        sabr::launch_sabr_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    const auto second = execute_diagonal<pg::SensitivityOrders::second>(
        prepare(selected, pg::SensitivityOrders::second), launch,
        sabr::launch_sabr_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::second
        >
    );
    const auto first = execute(
        prepare(selected, pg::SensitivityOrders::first),
        launch, sabr::launch_sabr_european_option_price_gradients_cuda<Side>
    );
    const auto price_only = execute(
        prepare({}, pg::SensitivityOrders::first),
        launch, sabr::launch_sabr_european_option_price_gradients_cuda<Side>
    );
    const pg::PriceGradientConfiguration maturity{{
        {"product.maturity_years", {1.0f/504.0f, pg::BumpScale::absolute}}
    }};
    const auto maturity_first = execute(
        prepare(maturity, pg::SensitivityOrders::first),
        launch, sabr::launch_sabr_european_option_price_gradients_cuda<Side>
    );
    const auto maturity_diagonal = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(maturity, pg::SensitivityOrders::first_and_second), launch,
        sabr::launch_sabr_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    DeviceArray<sabr::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    DeviceArray<float> canonical_prices(models.size());
    DeviceArray<float> canonical_errors(models.size());
    sabr::launch_sabr_european_option_cuda<Side>(
        device_models.data, models.size(), products.data(),
        device_products.data, products.size(), PriceConstruction::Aligned,
        models.size(), 0U, models.size(), paths, 1.0f/504.0f, 2U,
        128U, models.size(), 1709U, canonical_prices.data,
        canonical_errors.data
    );
    check_cuda(cudaDeviceSynchronize(), "SABR canonical central price");
    const auto reference_prices = canonical_prices.read();
    const auto reference_errors = canonical_errors.read();
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(combined.price[row], reference_prices[row],
             "SABR diagonal central price differs from price-only");
        same(combined.price_error[row], reference_errors[row],
             "SABR diagonal central error differs from price-only");
        same(combined.price[row], first.price[row],
             "SABR sensitivity order changed price");
        same(combined.price[row], second.price[row],
             "SABR second-only request changed price");
        same(combined.price[row], price_only.price[row],
             "SABR empty selection changed price");
        same(combined.price[row], maturity_first.price[row],
             "SABR maturity selection changed central price");
        same(combined.price[row], maturity_diagonal.price[row],
             "SABR maturity diagonal changed central price");
        require(std::isfinite(maturity_first.gradient[row]),
                "SABR maturity gradient is non-finite.");
        require(std::isfinite(maturity_diagonal.gradient[row])
                && std::isfinite(maturity_diagonal.diagonal_hessian[row]),
                "SABR maturity diagonal is non-finite.");
        for (std::size_t i = 0U; i < selected.sensitivities.size(); ++i) {
            const auto index = row*selected.sensitivities.size()+i;
            require(std::isfinite(combined.gradient[index])
                    && std::isfinite(combined.diagonal_hessian[index]),
                    "SABR sensitivity is non-finite.");
            same(combined.diagonal_hessian[index], second.diagonal_hessian[index],
                 "SABR Hessian changed with requested orders");
        }
    }
    const pg::PriceGradientConfiguration spot_only{{selected.sensitivities[0U]}};
    const auto solo = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        prepare(spot_only, pg::SensitivityOrders::first_and_second), launch,
        sabr::launch_sabr_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(combined.gradient[row*selected.sensitivities.size()],
             solo.gradient[row], "SABR gradient changed with selection");
        same(combined.diagonal_hessian[row*selected.sensitivities.size()],
             solo.diagonal_hessian[row], "SABR Hessian changed with selection");
    }
    const std::vector<sabr::ModelParameters> short_models{models[0U]};
    const std::vector<product::EuropeanOptionParameters> short_products{
        {1.0f, 1U}
    };
    const pg::PriceGradientConfiguration short_maturity{{
        {"product.maturity_years", {2.0f/504.0f, pg::BumpScale::absolute}}
    }};
    const auto short_plan = sabr::prepare_sabr_european_option_sensitivities(
        short_models, short_products, PriceConstruction::Aligned, {},
        short_maturity, {pg::SensitivityOrders::first_and_second}
    );
    const auto short_result = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        short_plan, launch,
        sabr::launch_sabr_european_option_diagonal_sensitivities_cuda<
            Side, pg::SensitivityOrders::first_and_second
        >
    );
    require(short_result.stencils[0U].kind == pg::StencilKind::forward
            && pg::active_node_count(short_result.stencils[0U]) == 4U
            && std::isfinite(short_result.diagonal_hessian[0U]),
            "SABR short-maturity four-node diagonal failed.");
}

int main(int argc, char** argv) {
    try {
        std::size_t paths = 1025U;
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") paths = 257U;
        else if (argc != 1) throw std::invalid_argument("Usage: test [--sanitizer]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        check<OptionSide::call>(paths);
        check<OptionSide::put>(paths);
        std::cout << "SABR first and diagonal second order: prices, CRN selection, boundaries passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
