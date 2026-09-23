// Compact-gradient coverage for the remaining finite-step equity diffusions.
#include "model/equity/markovian/heston_3_2/product/european_option_price_delta.cuh"
#include "model/equity/markovian/heston_3_2/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/schobel_zhu/product/european_option_price_delta.cuh"
#include "model/equity/markovian/schobel_zhu/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/stein_stein/product/european_option_price_delta.cuh"
#include "model/equity/markovian/stein_stein/product/european_option_price_gradients.cuh"
#include "cuda_test_support.cuh"
#include "diagonal_cuda_test_support.cuh"

#include <string_view>

using namespace price_gradient_test;
namespace h32 = model::equity::heston_3_2;
namespace sz = model::equity::schobel_zhu;
namespace ss = model::equity::stein_stein;

struct Heston32Case {
    using Model = h32::ModelParameters;
    using Plan = h32::EuropeanOptionPriceGradientPlan;
    static constexpr std::string_view name = "Heston 3/2";

    static std::vector<Model> models() {
        return {
            {.75f, .03f, .01f, .04f, 1.5f, .04f, .3f, -.7f},
            {1.2f, 0.f, 0.f, .06f, .8f, .05f, .4f, -.3f},
            {1.4f, .01f, 0.f, .03f, 1.f, .04f, .25f, 1.f},
        };
    }

    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {.005}},
            {"model.risk_free_rate", {.0001, pg::BumpScale::absolute}},
            {"model.dividend_yield", {.0001, pg::BumpScale::absolute}},
            {"model.initial_variance", {.001, pg::BumpScale::absolute}},
            {"model.mean_reversion", {.005}},
            {"model.long_run_variance", {.005}},
            {"model.volatility_of_variance", {.005}},
            {"model.rho", {.002, pg::BumpScale::absolute}},
            {"product.strike", {.002}},
            {"product.maturity_years", {1.f / 504.f, pg::BumpScale::absolute}},
        }};
    }

    static Plan prepare(
        const std::vector<Model>& models,
        const std::vector<product::EuropeanOptionParameters>& products,
        const pg::PriceGradientConfiguration& sensitivities,
        pg::SensitivityOrders orders
    ) {
        return h32::prepare_heston_3_2_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, sensitivities,
            {orders}
        );
    }

    template<OptionSide Side>
    static void launch_first(
        const Plan& plan,
        Plan::DeviceInputs inputs,
        Plan::StencilOutputs stencils,
        const pg::LaunchConfiguration& launch,
        pg::Outputs outputs
    ) {
        h32::launch_heston_3_2_european_option_price_gradients_cuda<Side>(
            plan, inputs, stencils, launch, outputs
        );
    }

    template<OptionSide Side, pg::SensitivityOrders Orders>
    static void launch_diagonal(
        const Plan& plan,
        Plan::DeviceInputs inputs,
        Plan::DiagonalStencilOutputs stencils,
        const pg::LaunchConfiguration& launch,
        pg::SensitivityOutputs outputs
    ) {
        h32::launch_heston_3_2_european_option_diagonal_sensitivities_cuda<
            Side, Orders
        >(plan, inputs, stencils, launch, outputs);
    }

    template<OptionSide Side>
    static void launch_legacy(
        const std::vector<Model>& models,
        const DeviceArray<Model>& device_models,
        const std::vector<product::EuropeanOptionParameters>& products,
        const DeviceArray<product::EuropeanOptionParameters>& device_products,
        std::size_t paths,
        float* outputs
    ) {
        h32::launch_heston_3_2_european_option_price_delta_cuda<Side>(
            models.data(), device_models.data, models.size(), products.data(),
            device_products.data, products.size(), PriceConstruction::Aligned,
            models.size(), 0U, models.size(), paths, 1.f / 504.f, 2U, 128U,
            models.size(), 2027U, {.01f}, outputs, outputs + models.size(),
            outputs + 2U * models.size(), outputs + 3U * models.size()
        );
    }
};

struct SchobelZhuCase {
    using Model = sz::ModelParameters;
    using Plan = sz::EuropeanOptionPriceGradientPlan;
    static constexpr std::string_view name = "Schobel-Zhu";

    static std::vector<Model> models() {
        return {
            {.75f, .03f, .01f, .2f, 1.5f, .18f, .3f, -.7f},
            {1.2f, 0.f, 0.f, .3f, .8f, .25f, .4f, -.3f},
            {1.4f, .01f, 0.f, .15f, 1.f, .2f, .25f, .999f},
        };
    }

    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {.005}},
            {"model.risk_free_rate", {.0001, pg::BumpScale::absolute}},
            {"model.dividend_yield", {.0001, pg::BumpScale::absolute}},
            {"model.initial_volatility", {.005}},
            {"model.mean_reversion", {.005}},
            {"model.long_run_volatility", {.005}},
            {"model.volatility_of_volatility", {.005}},
            {"model.correlation", {.002, pg::BumpScale::absolute}},
            {"product.strike", {.002}},
            {"product.maturity_years", {1.f / 504.f, pg::BumpScale::absolute}},
        }};
    }

    static Plan prepare(
        const std::vector<Model>& models,
        const std::vector<product::EuropeanOptionParameters>& products,
        const pg::PriceGradientConfiguration& sensitivities,
        pg::SensitivityOrders orders
    ) {
        return sz::prepare_schobel_zhu_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, sensitivities,
            {orders}
        );
    }

    template<OptionSide Side>
    static void launch_first(
        const Plan& plan,
        Plan::DeviceInputs inputs,
        Plan::StencilOutputs stencils,
        const pg::LaunchConfiguration& launch,
        pg::Outputs outputs
    ) {
        sz::launch_schobel_zhu_european_option_price_gradients_cuda<Side>(
            plan, inputs, stencils, launch, outputs
        );
    }

    template<OptionSide Side, pg::SensitivityOrders Orders>
    static void launch_diagonal(
        const Plan& plan,
        Plan::DeviceInputs inputs,
        Plan::DiagonalStencilOutputs stencils,
        const pg::LaunchConfiguration& launch,
        pg::SensitivityOutputs outputs
    ) {
        sz::launch_schobel_zhu_european_option_diagonal_sensitivities_cuda<
            Side, Orders
        >(plan, inputs, stencils, launch, outputs);
    }

    template<OptionSide Side>
    static void launch_legacy(
        const std::vector<Model>& models,
        const DeviceArray<Model>& device_models,
        const std::vector<product::EuropeanOptionParameters>& products,
        const DeviceArray<product::EuropeanOptionParameters>& device_products,
        std::size_t paths,
        float* outputs
    ) {
        sz::launch_schobel_zhu_european_option_price_delta_cuda<Side>(
            models.data(), device_models.data, models.size(), products.data(),
            device_products.data, products.size(), PriceConstruction::Aligned,
            models.size(), 0U, models.size(), paths, 1.f / 504.f, 2U, 128U,
            models.size(), 2027U, {.01f}, outputs, outputs + models.size(),
            outputs + 2U * models.size(), outputs + 3U * models.size()
        );
    }
};

struct SteinSteinCase {
    using Model = ss::ModelParameters;
    using Plan = ss::EuropeanOptionPriceGradientPlan;
    static constexpr std::string_view name = "Stein-Stein";

    static std::vector<Model> models() {
        return {
            {.75f, .03f, .01f, .2f, 1.5f, .3f, -.7f},
            {1.2f, 0.f, 0.f, .3f, .8f, .4f, -.3f},
            {1.4f, .01f, 0.f, .15f, 1.f, .25f, 1.f},
        };
    }

    static pg::PriceGradientConfiguration sensitivities() {
        return {{
            {"model.spot", {.005}},
            {"model.risk_free_rate", {.0001, pg::BumpScale::absolute}},
            {"model.dividend_yield", {.0001, pg::BumpScale::absolute}},
            {"model.initial_volatility", {.005}},
            {"model.mean_reversion", {.005}},
            {"model.volatility_of_volatility", {.005}},
            {"model.rho", {.002, pg::BumpScale::absolute}},
            {"product.strike", {.002}},
            {"product.maturity_years", {1.f / 504.f, pg::BumpScale::absolute}},
        }};
    }

    static Plan prepare(
        const std::vector<Model>& models,
        const std::vector<product::EuropeanOptionParameters>& products,
        const pg::PriceGradientConfiguration& sensitivities,
        pg::SensitivityOrders orders
    ) {
        return ss::prepare_stein_stein_european_option_sensitivities(
            models, products, PriceConstruction::Aligned, {}, sensitivities,
            {orders}
        );
    }

    template<OptionSide Side>
    static void launch_first(
        const Plan& plan,
        Plan::DeviceInputs inputs,
        Plan::StencilOutputs stencils,
        const pg::LaunchConfiguration& launch,
        pg::Outputs outputs
    ) {
        ss::launch_stein_stein_european_option_price_gradients_cuda<Side>(
            plan, inputs, stencils, launch, outputs
        );
    }

    template<OptionSide Side, pg::SensitivityOrders Orders>
    static void launch_diagonal(
        const Plan& plan,
        Plan::DeviceInputs inputs,
        Plan::DiagonalStencilOutputs stencils,
        const pg::LaunchConfiguration& launch,
        pg::SensitivityOutputs outputs
    ) {
        ss::launch_stein_stein_european_option_diagonal_sensitivities_cuda<
            Side, Orders
        >(plan, inputs, stencils, launch, outputs);
    }

    template<OptionSide Side>
    static void launch_legacy(
        const std::vector<Model>& models,
        const DeviceArray<Model>& device_models,
        const std::vector<product::EuropeanOptionParameters>& products,
        const DeviceArray<product::EuropeanOptionParameters>& device_products,
        std::size_t paths,
        float* outputs
    ) {
        ss::launch_stein_stein_european_option_price_delta_cuda<Side>(
            models.data(), device_models.data, models.size(), products.data(),
            device_products.data, products.size(), PriceConstruction::Aligned,
            models.size(), 0U, models.size(), paths, 1.f / 504.f, 2U, 128U,
            models.size(), 2027U, {.01f}, outputs, outputs + models.size(),
            outputs + 2U * models.size(), outputs + 3U * models.size()
        );
    }
};

template<typename Case, OptionSide Side>
void check_first_order(std::size_t paths) {
    const auto models = Case::models();
    const std::vector<product::EuropeanOptionParameters> products{
        {.8f, 16U}, {1.2f, 12U}, {1.1f, 8U},
    };
    const pg::PriceGradientConfiguration spot{{
        {"model.spot", {.005}},
    }};
    const auto launcher = [](const auto&... arguments) {
        Case::template launch_first<Side>(arguments...);
    };
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths, 128U,
        models.size(), 2027U, 1U,
    };
    const auto selected = execute(
        Case::prepare(models, products, spot, pg::SensitivityOrders::first),
        launch,
        launcher
    );
    const auto price_only = execute(
        Case::prepare(models, products, {}, pg::SensitivityOrders::first),
        launch,
        launcher
    );

    DeviceArray<typename Case::Model> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    DeviceArray<float> legacy(4U * models.size());
    Case::template launch_legacy<Side>(
        models, device_models, products, device_products, paths, legacy.data
    );
    check_cuda(cudaDeviceSynchronize(), "Legacy diffusion price-delta test");
    const auto reference = legacy.read();
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(selected.price[row], reference[row], "Diffusion price parity");
        same(
            selected.price_error[row], reference[models.size() + row],
            "Diffusion price-error parity"
        );
        same(
            selected.gradient[row], reference[2U * models.size() + row],
            "Diffusion delta parity"
        );
        same(
            selected.gradient_error[row],
            reference[3U * models.size() + row],
            "Diffusion delta-error parity"
        );
        same(
            price_only.price[row], selected.price[row],
            "Empty selection changed diffusion price"
        );
        same(
            price_only.price_error[row], selected.price_error[row],
            "Empty selection changed diffusion price error"
        );
    }
}

template<typename Case>
void check_diagonal(std::size_t paths) {
    const auto models = Case::models();
    const std::vector<product::EuropeanOptionParameters> products{
        {.8f, 16U}, {1.2f, 12U}, {1.1f, 8U},
    };
    const auto sensitivities = Case::sensitivities();
    pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, models.size(), paths, 128U,
        models.size() * sensitivities.sensitivities.size(), 2027U, 1U,
    };
    const auto combined = execute_diagonal<
        pg::SensitivityOrders::first_and_second
    >(
        Case::prepare(
            models,
            products,
            sensitivities,
            pg::SensitivityOrders::first_and_second
        ),
        launch,
        [](const auto&... arguments) {
            Case::template launch_diagonal<
                OptionSide::call,
                pg::SensitivityOrders::first_and_second
            >(arguments...);
        }
    );
    const auto second = execute_diagonal<pg::SensitivityOrders::second>(
        Case::prepare(
            models,
            products,
            sensitivities,
            pg::SensitivityOrders::second
        ),
        launch,
        [](const auto&... arguments) {
            Case::template launch_diagonal<
                OptionSide::call,
                pg::SensitivityOrders::second
            >(arguments...);
        }
    );
    for (std::size_t row = 0U; row < models.size(); ++row) {
        same(combined.price[row], second.price[row], "Order changed price");
        same(
            combined.price_error[row], second.price_error[row],
            "Order changed price error"
        );
        for (std::size_t sensitivity = 0U;
             sensitivity < sensitivities.sensitivities.size();
             ++sensitivity) {
            const auto index =
                row * sensitivities.sensitivities.size() + sensitivity;
            require(
                std::isfinite(combined.gradient[index])
                    && std::isfinite(combined.diagonal_hessian[index]),
                "Diffusion diagonal sensitivity is non-finite."
            );
            same(
                combined.diagonal_hessian[index],
                second.diagonal_hessian[index],
                "Order changed diagonal Hessian"
            );
            same(
                combined.diagonal_hessian_error[index],
                second.diagonal_hessian_error[index],
                "Order changed diagonal-Hessian error"
            );
        }
    }
}

template<typename Case>
void check_model(std::size_t paths) {
    check_first_order<Case, OptionSide::call>(paths);
    check_first_order<Case, OptionSide::put>(paths);
    check_diagonal<Case>(paths);
    std::cout << Case::name
              << ": legacy parity and first/diagonal-second sensitivities passed\n";
}

int main(int argc, char** argv) {
    try {
        std::size_t paths = 2053U;
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            paths = 257U;
        } else if (argc != 1) {
            throw std::invalid_argument("Usage: test [--sanitizer]");
        }
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            return 77;
        }
        check_model<Heston32Case>(paths);
        check_model<SchobelZhuCase>(paths);
        check_model<SteinSteinCase>(paths);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
