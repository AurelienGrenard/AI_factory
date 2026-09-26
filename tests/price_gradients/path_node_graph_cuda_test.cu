// Public path-product bindings: mono and node-graph sensitivities are bitwise equal.
#include "model/equity/markovian/bates/product/range_accrual_price_gradients.cuh"
#include "model/equity/markovian/heston/product/asian_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/athena_autocall_price_gradients.cuh"
#include "model/equity/markovian/merton/product/forward_start_option_price_gradients.cuh"
#include "product/asian_option/parameters.hpp"
#include "product/athena_autocall/parameters.hpp"
#include "product/forward_start_option/parameters.hpp"
#include "product/range_accrual/parameters.hpp"
#include "diagonal_cuda_test_support.cuh"

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

constexpr pg::SensitivityOrders orders =
    pg::SensitivityOrders::first_and_second;

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
        {orders}
    );
    price_gradient_test::require_mono_node_graph_parity(
        plan,
        heston::launch_heston_asian_option_diagonal_sensitivities_cuda<
            OptionSide::call, orders>,
        heston::heston_asian_option_node_graph_workspace_bytes<
            OptionSide::call, orders>,
        heston::launch_heston_asian_option_node_graph_sensitivities_cuda<
            OptionSide::call, orders>,
        paths,
        4101U,
        "heston.asian_option"
    );
}

void check_merton_athena(std::size_t paths) {
    const std::vector<merton::ModelParameters> models{{
        1.0f, 0.02f, 0.01f, 0.20f, 0.25f, -0.05f, 0.15f
    }};
    const std::vector<product::AthenaAutocallParameters> products{{
        12U, 3U, 1.0f, 0.70f, 0.08f
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.005f, pg::BumpScale::relative}},
        {"model.jump_intensity", {.05f, pg::BumpScale::absolute}},
        {"product.annual_coupon_rate", {.0005f, pg::BumpScale::absolute}},
    }};
    const auto plan = merton::prepare_merton_athena_autocall_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        {orders}
    );
    price_gradient_test::require_mono_node_graph_parity(
        plan,
        merton::launch_merton_athena_autocall_diagonal_sensitivities_cuda<
            orders>,
        merton::merton_athena_autocall_node_graph_workspace_bytes<orders>,
        merton::launch_merton_athena_autocall_node_graph_sensitivities_cuda<
            orders>,
        paths,
        4111U,
        "merton.athena_autocall"
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
            {orders}
        );
    price_gradient_test::require_mono_node_graph_parity(
        plan,
        merton::launch_merton_forward_start_option_diagonal_sensitivities_cuda<
            OptionSide::put, orders>,
        merton::merton_forward_start_option_node_graph_workspace_bytes<
            OptionSide::put, orders>,
        merton::launch_merton_forward_start_option_node_graph_sensitivities_cuda<
            OptionSide::put, orders>,
        paths,
        4121U,
        "merton.forward_start_option"
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
        {orders}
    );
    price_gradient_test::require_mono_node_graph_parity(
        plan,
        bates::launch_bates_range_accrual_diagonal_sensitivities_cuda<orders>,
        bates::bates_range_accrual_node_graph_workspace_bytes<orders>,
        bates::launch_bates_range_accrual_node_graph_sensitivities_cuda<orders>,
        paths,
        4131U,
        "bates.range_accrual"
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
        check_merton_athena(paths);
        check_merton_forward_start(paths);
        check_bates_range_accrual(paths);
        std::cout << "Dense, regular, calendar and jump path bindings are "
                     "bitwise equal between mono and node_graph\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
