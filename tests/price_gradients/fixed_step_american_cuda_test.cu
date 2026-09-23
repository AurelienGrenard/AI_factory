// Fixed-step American gradients share the same device-prepared LSM contract.
#include "model/equity/markovian/cev/product/american_option_price_delta.cuh"
#include "model/equity/markovian/cev/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/product/american_option_price_delta.cuh"
#include "model/equity/markovian/schobel_zhu/product/american_option_price_gradients.cuh"
#include "tests/price_gradients/american_cuda_test_support.cuh"

#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace american_test = price_gradient_test::american;
namespace cev = model::equity::cev;
namespace sz = model::equity::schobel_zhu;

std::size_t test_paths = 4096U;

struct CevContract {
    using Model = cev::ModelParameters;
    static constexpr const char* name = "CEV";

    static std::vector<Model> models() {
        return {{1.0f, 0.03f, 0.01f, 0.20f, 0.70f}};
    }

    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {0.005f}},
            {"model.risk_free_rate", {0.0001f, pg::BumpScale::absolute}},
            {"model.dividend_yield", {0.0001f, pg::BumpScale::absolute}},
            {"model.sigma", {0.005f}},
            {"model.beta", {0.002f, pg::BumpScale::absolute}},
            {"product.strike", {0.005f}},
        }};
    }

    static constexpr std::size_t coupled_coordinate = 4U;

    static auto prepare(
        const std::vector<Model>& models,
        const std::vector<product::AmericanOptionParameters>& products,
        pg::TimeConfiguration time,
        const pg::PriceGradientConfiguration& configuration,
        pg::SensitivityOrders orders
    ) {
        return cev::prepare_cev_american_option_sensitivities(
            models, products, PriceConstruction::Aligned, time,
            configuration, {orders}
        );
    }

    static auto first_launcher() {
        return [](
            const auto& plan,
            auto inputs,
            auto stencils,
            const auto& launch,
            pg::SensitivityOutputs outputs
        ) {
            return cev::launch_cev_american_option_price_gradients_cuda<
                OptionSide::put
            >(
                plan,
                inputs,
                stencils,
                launch,
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
        return cev::launch_cev_american_option_diagonal_sensitivities_cuda<
            OptionSide::put,
            pg::SensitivityOrders::first_and_second
        >;
    }

    static auto legacy_launcher() {
        return american_test::fixed_step_legacy_reference<
            cev::launch_cev_american_option_price_delta_cuda<
                OptionSide::put
            >
        >();
    }
};

struct SchobelZhuContract {
    using Model = sz::ModelParameters;
    static constexpr const char* name = "Schöbel–Zhu";

    static std::vector<Model> models() {
        return {{
            1.0f, 0.03f, 0.01f, 0.20f, 1.50f, 0.18f, 0.30f, -0.70f
        }};
    }

    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {0.005f}},
            {"model.risk_free_rate", {0.0001f, pg::BumpScale::absolute}},
            {"model.dividend_yield", {0.0001f, pg::BumpScale::absolute}},
            {"model.initial_volatility", {0.005f}},
            {"model.mean_reversion", {0.005f}},
            {"model.long_run_volatility", {0.005f}},
            {"model.volatility_of_volatility", {0.005f}},
            {"model.correlation", {0.002f, pg::BumpScale::absolute}},
            {"product.strike", {0.005f}},
        }};
    }

    static constexpr std::size_t coupled_coordinate = 7U;

    static auto prepare(
        const std::vector<Model>& models,
        const std::vector<product::AmericanOptionParameters>& products,
        pg::TimeConfiguration time,
        const pg::PriceGradientConfiguration& configuration,
        pg::SensitivityOrders orders
    ) {
        return sz::prepare_schobel_zhu_american_option_sensitivities(
            models, products, PriceConstruction::Aligned, time,
            configuration, {orders}
        );
    }

    static auto first_launcher() {
        return [](
            const auto& plan,
            auto inputs,
            auto stencils,
            const auto& launch,
            pg::SensitivityOutputs outputs
        ) {
            return sz::launch_schobel_zhu_american_option_price_gradients_cuda<
                OptionSide::put
            >(
                plan,
                inputs,
                stencils,
                launch,
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
        return sz::
            launch_schobel_zhu_american_option_diagonal_sensitivities_cuda<
                OptionSide::put,
                pg::SensitivityOrders::first_and_second
            >;
    }

    static auto legacy_launcher() {
        return american_test::fixed_step_legacy_reference<
            sz::launch_schobel_zhu_american_option_price_delta_cuda<
                OptionSide::put
            >
        >();
    }
};

template<typename Contract>
void verify() {
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
    std::cerr << "Checking " << Contract::name << " American sensitivities.\n";
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
                "Usage: test_price_gradients_fixed_step_american_cuda "
                "[--sanitizer]"
            );
        }
        verify<CevContract>();
        verify<SchobelZhuContract>();
        std::cout << "Fixed-step American sensitivities passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
