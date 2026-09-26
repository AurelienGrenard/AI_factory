// Public binding matrix: mono and node-graph terminal sensitivities are bitwise equal.
#include "model/equity/markovian/bates/dataset.hpp"
#include "model/equity/markovian/bates/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/bates/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/bates/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "model/equity/markovian/cev/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/cev/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/cev/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/heston/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "model/equity/markovian/heston_3_2/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/kou/dataset.hpp"
#include "model/equity/markovian/kou/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/kou/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/kou/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/dataset.hpp"
#include "model/equity/markovian/merton/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/dataset.hpp"
#include "model/equity/markovian/normal_inverse_gaussian/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "model/equity/markovian/sabr/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/dataset.hpp"
#include "model/equity/markovian/schobel_zhu/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "model/equity/markovian/stein_stein/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "model/equity/markovian/variance_gamma/product/asset_or_nothing_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/product/digital_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/product/european_option_price_gradients.cuh"
#include "product/asset_or_nothing_option/parameters.hpp"
#include "product/digital_option/parameters.hpp"
#include "product/european_option/parameters.hpp"
#include "diagonal_cuda_test_support.cuh"

#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ai_factory::workbench;
using price_gradient_test::DiagonalResults;
using price_gradient_test::require;
namespace pg = price_gradients;

#define CHECK_TERMINAL_MODEL(MODEL, MODEL_NAMESPACE, PATHS, SEED) do {     auto models = MODEL_NAMESPACE::load_models(         "datasets/model/equity/markovian/" #MODEL         "/parameters/" #MODEL "_01.json"     );     require(!models.empty(), "The model fixture is empty.");     models.resize(1U);     const pg::PriceGradientConfiguration selection{{         {"model.spot", {.005f}},         {"product.strike", {.005f}},     }};     const std::vector<product::EuropeanOptionParameters> european{{1.f, 8U}};     const auto european_plan =         MODEL_NAMESPACE::prepare_##MODEL##_european_option_sensitivities(             models, european, PriceConstruction::Aligned, {}, selection,             {pg::SensitivityOrders::first_and_second}         );     price_gradient_test::require_mono_node_graph_parity(         european_plan,         MODEL_NAMESPACE::launch_##MODEL##_european_option_diagonal_sensitivities_cuda<             OptionSide::call, pg::SensitivityOrders::first_and_second>,         MODEL_NAMESPACE::MODEL##_european_option_node_graph_workspace_bytes<             OptionSide::call, pg::SensitivityOrders::first_and_second>,         MODEL_NAMESPACE::launch_##MODEL##_european_option_node_graph_sensitivities_cuda<             OptionSide::call, pg::SensitivityOrders::first_and_second>,         PATHS, SEED, #MODEL ".european_option"     );     const std::vector<product::AssetOrNothingOptionParameters> asset{{1.f, 8U}};     const auto asset_plan =         MODEL_NAMESPACE::prepare_##MODEL##_asset_or_nothing_option_sensitivities(             models, asset, PriceConstruction::Aligned, {}, selection,             {pg::SensitivityOrders::first_and_second}         );     price_gradient_test::require_mono_node_graph_parity(         asset_plan,         MODEL_NAMESPACE::launch_##MODEL##_asset_or_nothing_option_diagonal_sensitivities_cuda<             OptionSide::put, pg::SensitivityOrders::first_and_second>,         MODEL_NAMESPACE::MODEL##_asset_or_nothing_option_node_graph_workspace_bytes<             OptionSide::put, pg::SensitivityOrders::first_and_second>,         MODEL_NAMESPACE::launch_##MODEL##_asset_or_nothing_option_node_graph_sensitivities_cuda<             OptionSide::put, pg::SensitivityOrders::first_and_second>,         PATHS, SEED + 1U, #MODEL ".asset_or_nothing_option"     );     const std::vector<product::DigitalOptionParameters> digital{{1.f, 8U, 2.f}};     const auto digital_plan =         MODEL_NAMESPACE::prepare_##MODEL##_digital_option_sensitivities(             models, digital, PriceConstruction::Aligned, {}, selection,             {pg::SensitivityOrders::first_and_second}         );     price_gradient_test::require_mono_node_graph_parity(         digital_plan,         MODEL_NAMESPACE::launch_##MODEL##_digital_option_diagonal_sensitivities_cuda<             OptionSide::call, pg::SensitivityOrders::first_and_second>,         MODEL_NAMESPACE::MODEL##_digital_option_node_graph_workspace_bytes<             OptionSide::call, pg::SensitivityOrders::first_and_second>,         MODEL_NAMESPACE::launch_##MODEL##_digital_option_node_graph_sensitivities_cuda<             OptionSide::call, pg::SensitivityOrders::first_and_second>,         PATHS, SEED + 2U, #MODEL ".digital_option"     ); } while (false)

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
        CHECK_TERMINAL_MODEL(bates, model::equity::bates, paths, 3001U);
        CHECK_TERMINAL_MODEL(cev, model::equity::cev, paths, 3011U);
        CHECK_TERMINAL_MODEL(heston, model::equity::heston, paths, 3021U);
        CHECK_TERMINAL_MODEL(heston_3_2, model::equity::heston_3_2, paths, 3031U);
        CHECK_TERMINAL_MODEL(kou, model::equity::kou, paths, 3041U);
        CHECK_TERMINAL_MODEL(merton, model::equity::merton, paths, 3051U);
        CHECK_TERMINAL_MODEL(normal_inverse_gaussian, model::equity::normal_inverse_gaussian, paths, 3061U);
        CHECK_TERMINAL_MODEL(sabr, model::equity::sabr, paths, 3071U);
        CHECK_TERMINAL_MODEL(schobel_zhu, model::equity::schobel_zhu, paths, 3081U);
        CHECK_TERMINAL_MODEL(stein_stein, model::equity::stein_stein, paths, 3091U);
        CHECK_TERMINAL_MODEL(variance_gamma, model::equity::variance_gamma, paths, 3101U);
        std::cout << "All 33 Markovian terminal MC bindings are bitwise equal "
                     "between mono and node_graph\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#undef CHECK_TERMINAL_MODEL
