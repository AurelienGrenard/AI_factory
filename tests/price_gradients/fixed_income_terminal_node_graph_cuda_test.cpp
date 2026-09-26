// Fixed-income terminal MC bindings: mono and node-graph bitwise parity.
#include "model/fixed_income/g2/product/european_swaption_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/product/nelson_siegel/european_swaption_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/product/svensson/european_swaption_price_gradients.cuh"
#include "tests/price_gradients/diagonal_cuda_test_support.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace g2 = model::fixed_income::g2;
namespace g2pp = model::fixed_income::g2_plus_plus;
namespace ns = model::fixed_income::g2_plus_plus::nelson_siegel;
namespace sv = model::fixed_income::g2_plus_plus::svensson;

constexpr pg::SensitivityOrders orders =
    pg::SensitivityOrders::first_and_second;

const std::vector<product::RegularEuropeanSwaptionParameters> products{{
    1.0f, 0.035f, 0.5f, 252U, 126U, 6U
}};

void check_g2(std::size_t paths) {
    const std::vector<g2::ModelParameters> models{{
        {0.15f, 0.010f, 0.45f, 0.015f, -0.35f},
        {0.020f, 0.010f},
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.volatility_x", {.0005f, pg::BumpScale::absolute}},
        {"model.initial_state_x", {.0005f, pg::BumpScale::absolute}},
        {"product.strike", {.0005f, pg::BumpScale::absolute}},
    }};
    const auto plan = g2::prepare_g2_european_swaption_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        {1.0f / 504.0f, 2U},
        selection,
        {orders}
    );
    price_gradient_test::require_mono_node_graph_parity(
        plan,
        g2::launch_g2_european_swaption_diagonal_sensitivities_cuda<
            SwaptionSide::payer, orders>,
        g2::g2_european_swaption_node_graph_workspace_bytes<
            SwaptionSide::payer, orders>,
        g2::launch_g2_european_swaption_node_graph_sensitivities_cuda<
            SwaptionSide::payer, orders>,
        paths,
        5101U,
        "g2.european_swaption"
    );
}

void check_g2_plus_plus_nelson_siegel(std::size_t paths) {
    const std::vector<g2pp::ModelParameters> models{{
        {0.15f, 0.010f, 0.45f, 0.015f, -0.35f}
    }};
    const std::vector<curve::nelson_siegel::NelsonSiegelParameters> curves{{
        0.030f, -0.015f, 0.020f, 2.0f
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.volatility_x", {.0005f, pg::BumpScale::absolute}},
        {"curve.beta0", {.0005f, pg::BumpScale::absolute}},
        {"product.strike", {.0005f, pg::BumpScale::absolute}},
    }};
    const auto plan =
        ns::prepare_g2_plus_plus_nelson_siegel_european_swaption_sensitivities(
            models,
            curves,
            products,
            PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection,
            {orders}
        );
    price_gradient_test::require_mono_node_graph_parity(
        plan,
        ns::launch_g2_plus_plus_nelson_siegel_european_swaption_diagonal_sensitivities_cuda<
            SwaptionSide::receiver, orders>,
        ns::g2_plus_plus_nelson_siegel_european_swaption_node_graph_workspace_bytes<
            SwaptionSide::receiver, orders>,
        ns::launch_g2_plus_plus_nelson_siegel_european_swaption_node_graph_sensitivities_cuda<
            SwaptionSide::receiver, orders>,
        paths,
        5111U,
        "g2_plus_plus.nelson_siegel.european_swaption"
    );
}

void check_g2_plus_plus_svensson(std::size_t paths) {
    const std::vector<g2pp::ModelParameters> models{{
        {0.15f, 0.010f, 0.45f, 0.015f, -0.35f}
    }};
    const std::vector<curve::svensson::SvenssonParameters> curves{{
        0.030f, -0.015f, 0.020f, -0.005f, 2.0f, 6.0f
    }};
    const pg::PriceGradientConfiguration selection{{
        {"model.volatility_y", {.0005f, pg::BumpScale::absolute}},
        {"curve.beta3", {.0005f, pg::BumpScale::absolute}},
        {"product.strike", {.0005f, pg::BumpScale::absolute}},
    }};
    const auto plan =
        sv::prepare_g2_plus_plus_svensson_european_swaption_sensitivities(
            models,
            curves,
            products,
            PriceConstruction::Aligned,
            {1.0f / 504.0f, 2U},
            selection,
            {orders}
        );
    price_gradient_test::require_mono_node_graph_parity(
        plan,
        sv::launch_g2_plus_plus_svensson_european_swaption_diagonal_sensitivities_cuda<
            SwaptionSide::payer, orders>,
        sv::g2_plus_plus_svensson_european_swaption_node_graph_workspace_bytes<
            SwaptionSide::payer, orders>,
        sv::launch_g2_plus_plus_svensson_european_swaption_node_graph_sensitivities_cuda<
            SwaptionSide::payer, orders>,
        paths,
        5121U,
        "g2_plus_plus.svensson.european_swaption"
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
        check_g2(paths);
        check_g2_plus_plus_nelson_siegel(paths);
        check_g2_plus_plus_svensson(paths);
        std::cout << "G2 and G2++ terminal MC bindings are bitwise equal "
                     "between mono and node_graph\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
