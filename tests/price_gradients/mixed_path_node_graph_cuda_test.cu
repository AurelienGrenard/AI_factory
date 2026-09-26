// Full-Hessian path bindings preserve existing price and diagonal results.
#include "model/equity/markovian/bates/product/cliquet_price_gradients.cuh"
#include "model/equity/markovian/bates/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/heston/product/asian_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/forward_start_option_price_gradients.cuh"
#include "product/asian_option/parameters.hpp"
#include "product/cliquet/parameters.hpp"
#include "product/forward_start_option/parameters.hpp"
#include "product/range_accrual/parameters.hpp"
#include "mixed_node_graph_cuda_test_support.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace bates = model::equity::bates;
namespace heston = model::equity::heston;
namespace merton = model::equity::merton;

constexpr auto orders = pg::SensitivityOrders::first_and_second;

void check_heston_asian(std::size_t paths) {
    const std::vector<heston::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.7f
    }};
    const std::vector<product::AsianOptionParameters> products{{1.0f, 8U}};
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f, pg::BumpScale::relative}},
        {"model.initial_variance", {.001f, pg::BumpScale::absolute}},
        {"product.strike", {.005f, pg::BumpScale::relative}},
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
