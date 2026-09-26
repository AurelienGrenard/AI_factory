// Exact-transition American gradients share one frozen-exercise contract.
#include "model/equity/markovian/black_scholes/product/american_option_price_delta.cuh"
#include "model/equity/markovian/black_scholes/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/kou/product/american_option_price_delta.cuh"
#include "model/equity/markovian/kou/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/american_option_price_delta.cuh"
#include "model/equity/markovian/merton/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/american_option_price_delta.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/variance_gamma/product/american_option_price_delta.cuh"
#include "model/equity/markovian/variance_gamma/product/american_option_price_gradients.cuh"
#include "tests/price_gradients/american_cuda_test_support.cuh"

#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace american_test = price_gradient_test::american;
namespace bs = model::equity::black_scholes;
namespace kou = model::equity::kou;
namespace merton = model::equity::merton;
namespace nig = model::equity::normal_inverse_gaussian;
namespace vg = model::equity::variance_gamma;

std::size_t test_paths = 4096U;

template<typename NamespaceTag>
struct ModelContract;

struct BlackScholesTag {};
struct MertonTag {};
struct KouTag {};
struct VarianceGammaTag {};
struct NormalInverseGaussianTag {};

template<>
struct ModelContract<BlackScholesTag> {
    using Model = bs::ModelParameters;
    static constexpr const char* name = "Black-Scholes";
    static std::vector<Model> models() {
        return {{1.0f, 0.03f, 0.01f, 0.20f}};
    }
    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {0.005f}},
            {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
            {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
            {"model.volatility", {0.005f}},
            {"product.strike", {0.005f}},
        }};
    }
    static constexpr std::size_t coupled_coordinate = 3U;
    static auto prepare(
        const std::vector<Model>& models,
        const std::vector<product::AmericanOptionParameters>& products,
        pg::TimeConfiguration time,
        const pg::PriceGradientConfiguration& configuration,
        pg::SensitivityOrders orders
    ) {
        return bs::prepare_black_scholes_american_option_sensitivities(
            models, products, PriceConstruction::Aligned, time,
            configuration, {orders}
        );
    }
    static auto first_launcher() {
        return [](
            const auto& plan, auto inputs, auto stencils,
            const auto& launch, pg::SensitivityOutputs outputs
        ) {
            return bs::launch_black_scholes_american_option_price_gradients_cuda<
                OptionSide::put
            >(
                plan, inputs, stencils, launch,
                {
                    outputs.prices,
                    outputs.price_standard_errors,
                    outputs.gradients,
                    outputs.gradient_standard_errors,
                    outputs.price_capacity,
                    outputs.sensitivity_capacity,
                }
            );
        };
    }
    static auto diagonal_launcher() {
        return bs::
            launch_black_scholes_american_option_diagonal_sensitivities_cuda<
                OptionSide::put,
                pg::SensitivityOrders::first_and_second
            >;
    }

    static auto node_graph_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            const auto bytes =
                bs::black_scholes_american_option_node_graph_workspace_bytes<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(plan, launch);
            price_gradient_test::DeviceArray<std::uint8_t> workspace(bytes);
            return bs::
                launch_black_scholes_american_option_node_graph_sensitivities_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    plan, inputs, stencils, launch, outputs,
                    workspace.data, workspace.count
                );
        };
    }

    static auto legacy_launcher() {
        return american_test::exact_transition_legacy_reference<
            bs::launch_black_scholes_american_option_price_delta_cuda<
                OptionSide::put
            >
        >();
    }
};

template<>
struct ModelContract<MertonTag> {
    using Model = merton::ModelParameters;
    static constexpr const char* name = "Merton";
    static std::vector<Model> models() {
        return {{1.0f, 0.03f, 0.01f, 0.20f, 0.50f, -0.10f, 0.25f}};
    }
    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {0.005f}},
            {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
            {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
            {"model.volatility", {0.005f}},
            {"model.jump_intensity", {0.05f, pg::BumpScale::absolute}},
            {"model.jump_log_mean", {0.002f, pg::BumpScale::absolute}},
            {"model.jump_log_volatility", {0.005f}},
            {"product.strike", {0.005f}},
        }};
    }
    static constexpr std::size_t coupled_coordinate = 4U;
    static auto prepare(const std::vector<Model>& models,
                        const std::vector<product::AmericanOptionParameters>& products,
                        pg::TimeConfiguration time,
                        const pg::PriceGradientConfiguration& configuration,
                        pg::SensitivityOrders orders) {
        return merton::prepare_merton_american_option_sensitivities(
            models, products, PriceConstruction::Aligned, time,
            configuration, {orders}
        );
    }
    static auto first_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            return merton::launch_merton_american_option_price_gradients_cuda<
                OptionSide::put
            >(plan, inputs, stencils, launch,
              {outputs.prices, outputs.price_standard_errors,
               outputs.gradients, outputs.gradient_standard_errors,
               outputs.price_capacity, outputs.sensitivity_capacity});
        };
    }
    static auto diagonal_launcher() {
        return merton::launch_merton_american_option_diagonal_sensitivities_cuda<
            OptionSide::put, pg::SensitivityOrders::first_and_second
        >;
    }

    static auto node_graph_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            const auto bytes =
                merton::merton_american_option_node_graph_workspace_bytes<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(plan, launch);
            price_gradient_test::DeviceArray<std::uint8_t> workspace(bytes);
            return merton::
                launch_merton_american_option_node_graph_sensitivities_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    plan, inputs, stencils, launch, outputs,
                    workspace.data, workspace.count
                );
        };
    }

    static auto legacy_launcher() {
        return american_test::exact_transition_legacy_reference<
            merton::launch_merton_american_option_price_delta_cuda<
                OptionSide::put
            >
        >();
    }
};

template<>
struct ModelContract<KouTag> {
    using Model = kou::ModelParameters;
    static constexpr const char* name = "Kou";
    static std::vector<Model> models() {
        return {{1.0f, 0.03f, 0.01f, 0.20f, 0.50f, 0.40f, 4.0f, 5.0f}};
    }
    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {0.005f}},
            {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
            {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
            {"model.volatility", {0.005f}},
            {"model.jump_intensity", {0.05f, pg::BumpScale::absolute}},
            {"model.up_probability", {0.002f, pg::BumpScale::absolute}},
            {"model.positive_jump_rate", {0.005f}},
            {"model.negative_jump_rate", {0.005f}},
            {"product.strike", {0.005f}},
        }};
    }
    static constexpr std::size_t coupled_coordinate = 4U;
    static auto prepare(const std::vector<Model>& models,
                        const std::vector<product::AmericanOptionParameters>& products,
                        pg::TimeConfiguration time,
                        const pg::PriceGradientConfiguration& configuration,
                        pg::SensitivityOrders orders) {
        return kou::prepare_kou_american_option_sensitivities(
            models, products, PriceConstruction::Aligned, time,
            configuration, {orders}
        );
    }
    static auto first_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            return kou::launch_kou_american_option_price_gradients_cuda<
                OptionSide::put
            >(plan, inputs, stencils, launch,
              {outputs.prices, outputs.price_standard_errors,
               outputs.gradients, outputs.gradient_standard_errors,
               outputs.price_capacity, outputs.sensitivity_capacity});
        };
    }
    static auto diagonal_launcher() {
        return kou::launch_kou_american_option_diagonal_sensitivities_cuda<
            OptionSide::put, pg::SensitivityOrders::first_and_second
        >;
    }

    static auto node_graph_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            const auto bytes =
                kou::kou_american_option_node_graph_workspace_bytes<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(plan, launch);
            price_gradient_test::DeviceArray<std::uint8_t> workspace(bytes);
            return kou::
                launch_kou_american_option_node_graph_sensitivities_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    plan, inputs, stencils, launch, outputs,
                    workspace.data, workspace.count
                );
        };
    }

    static auto legacy_launcher() {
        return american_test::exact_transition_legacy_reference<
            kou::launch_kou_american_option_price_delta_cuda<
                OptionSide::put
            >
        >();
    }
};

template<>
struct ModelContract<VarianceGammaTag> {
    using Model = vg::ModelParameters;
    static constexpr const char* name = "Variance-Gamma";
    static std::vector<Model> models() {
        return {{1.0f, 0.03f, 0.01f, 0.20f, 0.20f, -0.10f}};
    }
    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {0.005f}},
            {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
            {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
            {"model.sigma", {0.005f}},
            {"model.nu", {0.005f}},
            {"model.theta", {0.002f, pg::BumpScale::absolute}},
            {"product.strike", {0.005f}},
        }};
    }
    static constexpr std::size_t coupled_coordinate = 4U;
    static auto prepare(const std::vector<Model>& models,
                        const std::vector<product::AmericanOptionParameters>& products,
                        pg::TimeConfiguration time,
                        const pg::PriceGradientConfiguration& configuration,
                        pg::SensitivityOrders orders) {
        return vg::prepare_variance_gamma_american_option_sensitivities(
            models, products, PriceConstruction::Aligned, time,
            configuration, {orders}
        );
    }
    static auto first_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            return vg::launch_variance_gamma_american_option_price_gradients_cuda<
                OptionSide::put
            >(plan, inputs, stencils, launch,
              {outputs.prices, outputs.price_standard_errors,
               outputs.gradients, outputs.gradient_standard_errors,
               outputs.price_capacity, outputs.sensitivity_capacity});
        };
    }
    static auto diagonal_launcher() {
        return vg::launch_variance_gamma_american_option_diagonal_sensitivities_cuda<
            OptionSide::put, pg::SensitivityOrders::first_and_second
        >;
    }

    static auto node_graph_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            const auto bytes =
                vg::variance_gamma_american_option_node_graph_workspace_bytes<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(plan, launch);
            price_gradient_test::DeviceArray<std::uint8_t> workspace(bytes);
            return vg::
                launch_variance_gamma_american_option_node_graph_sensitivities_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    plan, inputs, stencils, launch, outputs,
                    workspace.data, workspace.count
                );
        };
    }

    static auto legacy_launcher() {
        return american_test::exact_transition_legacy_reference<
            vg::launch_variance_gamma_american_option_price_delta_cuda<
                OptionSide::put
            >
        >();
    }
};

template<>
struct ModelContract<NormalInverseGaussianTag> {
    using Model = nig::ModelParameters;
    static constexpr const char* name = "NIG";
    static std::vector<Model> models() {
        return {{1.0f, 0.03f, 0.01f, 8.0f, -2.0f, 0.50f}};
    }
    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {0.005f}},
            {"model.risk_free_rate", {0.0005f, pg::BumpScale::absolute}},
            {"model.dividend_yield", {0.0005f, pg::BumpScale::absolute}},
            {"model.alpha", {0.005f}},
            {"model.beta", {0.002f, pg::BumpScale::absolute}},
            {"model.delta", {0.005f}},
            {"product.strike", {0.005f}},
        }};
    }
    static constexpr std::size_t coupled_coordinate = 3U;
    static auto prepare(const std::vector<Model>& models,
                        const std::vector<product::AmericanOptionParameters>& products,
                        pg::TimeConfiguration time,
                        const pg::PriceGradientConfiguration& configuration,
                        pg::SensitivityOrders orders) {
        return nig::prepare_normal_inverse_gaussian_american_option_sensitivities(
            models, products, PriceConstruction::Aligned, time,
            configuration, {orders}
        );
    }
    static auto first_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            return nig::launch_normal_inverse_gaussian_american_option_price_gradients_cuda<
                OptionSide::put
            >(plan, inputs, stencils, launch,
              {outputs.prices, outputs.price_standard_errors,
               outputs.gradients, outputs.gradient_standard_errors,
               outputs.price_capacity, outputs.sensitivity_capacity});
        };
    }
    static auto diagonal_launcher() {
        return nig::launch_normal_inverse_gaussian_american_option_diagonal_sensitivities_cuda<
            OptionSide::put, pg::SensitivityOrders::first_and_second
        >;
    }

    static auto node_graph_launcher() {
        return [](const auto& plan, auto inputs, auto stencils,
                  const auto& launch, pg::SensitivityOutputs outputs) {
            const auto bytes =
                nig::normal_inverse_gaussian_american_option_node_graph_workspace_bytes<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(plan, launch);
            price_gradient_test::DeviceArray<std::uint8_t> workspace(bytes);
            return nig::
                launch_normal_inverse_gaussian_american_option_node_graph_sensitivities_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    plan, inputs, stencils, launch, outputs,
                    workspace.data, workspace.count
                );
        };
    }

    static auto legacy_launcher() {
        return american_test::exact_transition_legacy_reference<
            nig::launch_normal_inverse_gaussian_american_option_price_delta_cuda<
                OptionSide::put
            >
        >();
    }
};

template<typename Tag>
void verify() {
    using Contract = ModelContract<Tag>;
    std::cerr << "Checking " << Contract::name << " American sensitivities.\n";
    const auto models = Contract::models();
    const std::vector<product::AmericanOptionParameters> products{
        {1.0f, 63U, 7U}
    };
    const pg::TimeConfiguration time{1.0f / 504.0f, 2U};
    const auto sensitivities = Contract::sensitivities();
    const auto prepare = [&](const pg::PriceGradientConfiguration& selection,
                             pg::SensitivityOrders orders) {
        return Contract::prepare(models, products, time, selection, orders);
    };
    american_test::verify_model(
        Contract::name,
        models,
        products,
        time,
        sensitivities,
        Contract::coupled_coordinate,
        test_paths,
        prepare,
        Contract::first_launcher(),
        Contract::diagonal_launcher(),
        Contract::node_graph_launcher(),
        Contract::legacy_launcher()
    );
}

}  // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            test_paths = 257U;
        } else if (argc != 1) {
            throw std::invalid_argument(
                "Usage: test_price_gradients_exact_transition_american_cuda "
                "[--sanitizer]"
            );
        }
        verify<BlackScholesTag>();
        verify<MertonTag>();
        verify<KouTag>();
        verify<VarianceGammaTag>();
        verify<NormalInverseGaussianTag>();
        std::cout << "Exact-transition American sensitivities passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
