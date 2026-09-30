// Full-Hessian path bindings preserve existing price and diagonal results.
#include "model/equity/markovian/black_scholes/product/asian_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/geometric_asian_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/bates/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/bates/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/heston/product/asian_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/forward_start_option_price_gradients.cuh"
#include "product/asian_option/parameters.hpp"
#include "product/cliquet/parameters.hpp"
#include "product/forward_start_option/parameters.hpp"
#include "product/geometric_asian_option/parameters.hpp"
#include "product/range_accrual/parameters.hpp"
#include "mixed_node_graph_cuda_test_support.cuh"
#include "closed_form_mixed_cuda_test_support.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace bs = model::equity::black_scholes;
namespace bates = model::equity::bates;
namespace heston = model::equity::heston;
namespace merton = model::equity::merton;

constexpr auto orders = pg::SensitivityOrders::first_and_second;


template<
    typename Plan,
    typename DiagonalLauncher,
    typename WorkspaceSizer,
    typename MixedLauncher
>
void check_closed_form_terminal_maturity(
    const Plan& plan,
    DiagonalLauncher diagonal_launcher,
    WorkspaceSizer workspace_size,
    MixedLauncher mixed_launcher,
    const char* label
) {
    pg::LaunchConfiguration launch{
        pg::PricingMethod::closed_form,
        0U,
        plan.result_count,
        0U,
        128U,
        plan.sensitivity_count(),
        0U,
    };
    const auto diagonal = price_gradient_test::execute_diagonal<orders>(
        plan, launch, diagonal_launcher
    );
    const auto mixed = price_gradient_test::execute_closed_form_mixed(
        plan, launch, workspace_size, mixed_launcher, label
    );
    price_gradient_test::require_closed_form_diagonal_parity(
        diagonal, mixed, label
    );
    if (!(diagonal.gradient.size() == 2U
          && diagonal.diagonal_hessian.size() == 2U
          && mixed.mixed_hessians.size() == 1U
          && std::isfinite(diagonal.gradient[1U])
          && std::isfinite(diagonal.diagonal_hessian[1U])
          && std::abs(diagonal.gradient[1U]) > 1.0e-7f)) {
        throw std::runtime_error(
            std::string(label) + " did not evaluate terminal maturity."
        );
    }
}

void check_black_scholes_closed_form_paths() {
    const std::vector<bs::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.20f
    }};
    const auto selection = [](const char* coordinate, float displacement,
                              pg::BumpScale scale) {
        return pg::PriceGradientConfiguration{{
            {coordinate, {displacement, scale}},
            {"product.maturity_years",
             {1.0f / 504.0f, pg::BumpScale::absolute}},
        }};
    };

    const std::vector<product::ForwardStartOptionParameters> forward{{
        0.95f, 4U, 12U
    }};
    const auto forward_plan =
        bs::prepare_black_scholes_forward_start_option_sensitivities(
            models, forward, PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection("product.moneyness", .005f, pg::BumpScale::relative),
            pg::SensitivityRequest::full_hessian()
        );
    check_closed_form_terminal_maturity(
        forward_plan,
        [](const auto& plan, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            bs::launch_black_scholes_forward_start_option_diagonal_sensitivities_cuda<
                OptionSide::call, orders
            >(plan, inputs, stencils, launch, outputs);
        },
        [](const auto& plan, const auto& launch) {
            return bs::
                black_scholes_forward_start_option_mixed_node_graph_workspace_bytes<
                    OptionSide::call
                >(plan, launch);
        },
        [](const auto& plan, auto inputs, auto stencils,
           auto mixed_stencils, const auto& launch, auto outputs,
           auto mixed_outputs, void* workspace, std::size_t bytes) {
            bs::
                launch_black_scholes_forward_start_option_mixed_node_graph_sensitivities_cuda<
                    OptionSide::call
                >(plan, inputs, stencils, mixed_stencils, launch, outputs,
                  mixed_outputs, workspace, bytes);
        },
        "black_scholes.forward_start closed form"
    );

    const std::vector<product::GeometricAsianOptionParameters> geometric{{
        1.0f, 12U
    }};
    const auto geometric_plan =
        bs::prepare_black_scholes_geometric_asian_option_sensitivities(
            models, geometric, PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection("product.strike", .005f, pg::BumpScale::relative),
            pg::SensitivityRequest::full_hessian()
        );
    check_closed_form_terminal_maturity(
        geometric_plan,
        [](const auto& plan, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            bs::launch_black_scholes_geometric_asian_option_diagonal_sensitivities_cuda<
                OptionSide::put, orders
            >(plan, inputs, stencils, launch, outputs);
        },
        [](const auto& plan, const auto& launch) {
            return bs::
                black_scholes_geometric_asian_option_mixed_node_graph_workspace_bytes<
                    OptionSide::put
                >(plan, launch);
        },
        [](const auto& plan, auto inputs, auto stencils,
           auto mixed_stencils, const auto& launch, auto outputs,
           auto mixed_outputs, void* workspace, std::size_t bytes) {
            bs::
                launch_black_scholes_geometric_asian_option_mixed_node_graph_sensitivities_cuda<
                    OptionSide::put
                >(plan, inputs, stencils, mixed_stencils, launch, outputs,
                  mixed_outputs, workspace, bytes);
        },
        "black_scholes.geometric_asian closed form"
    );

    const std::vector<product::RangeAccrualParameters> accruals{{
        12U, 3U, 0.80f, 1.20f, 0.06f
    }};
    const auto accrual_plan =
        bs::prepare_black_scholes_range_accrual_sensitivities(
            models, accruals, PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection(
                "product.coupon_rate", .0005f, pg::BumpScale::absolute
            ),
            pg::SensitivityRequest::full_hessian()
        );
    check_closed_form_terminal_maturity(
        accrual_plan,
        [](const auto& plan, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            bs::launch_black_scholes_range_accrual_diagonal_sensitivities_cuda<
                orders
            >(plan, inputs, stencils, launch, outputs);
        },
        bs::black_scholes_range_accrual_mixed_node_graph_workspace_bytes,
        bs::launch_black_scholes_range_accrual_mixed_node_graph_sensitivities_cuda,
        "black_scholes.range_accrual closed form"
    );
}

void check_black_scholes_asian(std::size_t paths) {
    const std::vector<bs::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.20f
    }};
    const std::vector<product::AsianOptionParameters> products{{
        1.0f, 12U
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f, pg::BumpScale::relative}},
        {"model.volatility", {.005f, pg::BumpScale::relative}},
        {"product.strike", {.005f, pg::BumpScale::relative}},
        {"product.maturity_years",
         {1.0f / 504.0f, pg::BumpScale::absolute}},
    }};
    const auto plan =
        bs::prepare_black_scholes_asian_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection,
            pg::SensitivityRequest::full_hessian()
        );
    price_gradient_test::require_mixed_node_graph_parity(
        plan,
        bs::launch_black_scholes_asian_option_diagonal_sensitivities_cuda<
            OptionSide::call, orders>,
        bs::black_scholes_asian_option_mixed_node_graph_workspace_bytes<
            OptionSide::call>,
        bs::launch_black_scholes_asian_option_mixed_node_graph_sensitivities_cuda<
            OptionSide::call>,
        paths,
        5091U,
        "black_scholes.asian_option mixed graph"
    );
}

void check_heston_asian(std::size_t paths) {
    const std::vector<heston::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.7f
    }};
    const std::vector<product::AsianOptionParameters> products{{1.0f, 8U}};
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f, pg::BumpScale::relative}},
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"product.strike", {.005f, pg::BumpScale::relative}},
        {"product.maturity_years",
         {1.0f / 504.0f, pg::BumpScale::absolute}},
    }};
    const auto plan = heston::prepare_heston_asian_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        pg::SensitivityRequest::full_hessian()
    );
    price_gradient_test::require_mixed_node_graph_parity(
        plan,
        heston::launch_heston_asian_option_diagonal_sensitivities_cuda<
            OptionSide::call, orders>,
        heston::heston_asian_option_mixed_node_graph_workspace_bytes<
            OptionSide::call>,
        heston::launch_heston_asian_option_mixed_node_graph_sensitivities_cuda<
            OptionSide::call>,
        paths,
        5101U,
        "heston.asian_option mixed graph"
    );
}

void check_merton_forward_start(std::size_t paths) {
    const std::vector<merton::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.20f, 0.25f, -0.05f, 0.15f
    }};
    const std::vector<product::ForwardStartOptionParameters> products{{
        0.95f, 4U, 12U
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f, pg::BumpScale::relative}},
        {"model.jump_log_mean", {.002f, pg::BumpScale::absolute}},
        {"product.moneyness", {.005f, pg::BumpScale::relative}},
        {"product.maturity_years",
         {1.0f / 504.0f, pg::BumpScale::absolute}},
    }};
    const auto plan =
        merton::prepare_merton_forward_start_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection,
            pg::SensitivityRequest::full_hessian()
        );
    price_gradient_test::require_mixed_node_graph_parity(
        plan,
        merton::launch_merton_forward_start_option_diagonal_sensitivities_cuda<
            OptionSide::put, orders>,
        merton::merton_forward_start_option_mixed_node_graph_workspace_bytes<
            OptionSide::put>,
        merton::launch_merton_forward_start_option_mixed_node_graph_sensitivities_cuda<
            OptionSide::put>,
        paths,
        5111U,
        "merton.forward_start_option mixed graph"
    );
}

void check_bates_range_accrual(std::size_t paths) {
    const std::vector<bates::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.7f,
        0.25f, -0.05f, 0.15f
    }};
    const std::vector<product::RangeAccrualParameters> products{{
        12U, 3U, 0.80f, 1.20f, 0.06f
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f, pg::BumpScale::relative}},
        {"model.jump_intensity", {.05f, pg::BumpScale::absolute}},
        {"product.lower_barrier", {.005f, pg::BumpScale::relative}},
        {"product.maturity_years",
         {1.0f / 504.0f, pg::BumpScale::absolute}},
    }};
    const auto plan = bates::prepare_bates_range_accrual_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        pg::SensitivityRequest::full_hessian()
    );
    price_gradient_test::require_mixed_node_graph_parity(
        plan,
        bates::launch_bates_range_accrual_diagonal_sensitivities_cuda<orders>,
        bates::bates_range_accrual_mixed_node_graph_workspace_bytes,
        bates::launch_bates_range_accrual_mixed_node_graph_sensitivities_cuda,
        paths,
        5121U,
        "bates.range_accrual mixed graph"
    );
}


void check_bates_maximum_graph(std::size_t paths) {
    const std::vector<bates::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.7f,
        0.25f, -0.05f, 0.15f
    }};
    const std::vector<product::CliquetParameters> products{{
        12U, 3U, 1.0f, -0.10f, 0.10f, 0.0f, 0.30f
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f}},
        {"model.risk_free_rate", {.0005f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {.0005f, pg::BumpScale::absolute}},
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"model.kappa", {.005f}},
        {"model.theta", {.005f}},
        {"model.gamma", {.005f}},
        {"model.rho", {.002f, pg::BumpScale::absolute}},
        {"model.jump_intensity", {.05f, pg::BumpScale::absolute}},
        {"model.jump_log_mean", {.002f, pg::BumpScale::absolute}},
        {"model.jump_log_volatility", {.005f}},
        {"product.participation_rate", {.005f}},
        {"product.local_floor", {.001f, pg::BumpScale::absolute}},
        {"product.local_cap", {.001f, pg::BumpScale::absolute}},
        {"product.global_floor", {.001f, pg::BumpScale::absolute}},
        {"product.global_cap", {.001f, pg::BumpScale::absolute}},
    }};
    const auto plan = bates::prepare_bates_cliquet_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        pg::SensitivityRequest::full_hessian()
    );
    price_gradient_test::require(
        plan.sensitivity_graph.node_capacity == 529U,
        "Bates cliquet did not retain the 529-node capacity contract."
    );
    price_gradient_test::require_mixed_node_graph_parity(
        plan,
        bates::launch_bates_cliquet_diagonal_sensitivities_cuda<orders>,
        bates::bates_cliquet_mixed_node_graph_workspace_bytes,
        bates::launch_bates_cliquet_mixed_node_graph_sensitivities_cuda,
        paths,
        5131U,
        "bates.cliquet maximum mixed graph"
    );
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::size_t paths = 1025U;
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            paths = 257U;
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: test [--sanitizer]");
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        check_black_scholes_closed_form_paths();
        check_black_scholes_asian(paths);
        check_heston_asian(paths);
        check_merton_forward_start(paths);
        check_bates_range_accrual(paths);
        check_bates_maximum_graph(257U);
        std::cout << "Dense, exact-calendar and jump path full Hessians "
                     "preserve existing diagonal bits\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
