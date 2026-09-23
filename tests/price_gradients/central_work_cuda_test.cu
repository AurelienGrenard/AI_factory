// Count the work performed by the compact (row, sensitivity) terminal kernel.
#include "common/equity/price_gradients/terminal_device_prepared_launcher.cuh"
#include "common/equity/price_gradients/terminal_device_prepared_plan.hpp"
#include "common/equity/path_product_policy.cuh"
#include "tests/price_gradients/cuda_test_support.cuh"

#include <array>
#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>

using namespace ai_factory::workbench;
namespace epg = equity::price_gradients;
namespace pg = price_gradients;
using price_gradient_test::DeviceArray;
using price_gradient_test::require;

namespace {

__device__ unsigned int work_counts[3];

struct Model {
    float spot;
    float coefficient;
    float boundary_coefficient;
};

struct Product {
    float strike;
    std::uint32_t maturity_days;
};

struct ModelPreparation {
    using Model = ::Model;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = false;
    static constexpr bool kSupportsMaturityDiagonal = false;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.coefficient"},
        std::string_view{"model.boundary_coefficient"},
    };

    __host__ __device__ static bool valid(const Model& model) {
        return ::isfinite(model.spot) && model.spot > 0.0f
            && ::isfinite(model.coefficient)
            && ::isfinite(model.boundary_coefficient)
            && model.boundary_coefficient >= 0.0f;
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Model& model
    ) {
        switch (index) {
        case 0U: return model.spot;
        case 1U: return model.coefficient;
        case 2U: return model.boundary_coefficient;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Model& model,
        float value
    ) {
        switch (index) {
        case 0U: model.spot = value; break;
        case 1U: model.coefficient = value; break;
        case 2U: model.boundary_coefficient = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(
        const Model& first,
        const Model& second
    ) {
        return first.coefficient == second.coefficient
            && first.boundary_coefficient == second.boundary_coefficient;
    }
};

struct ProductPreparation {
    using Product = ::Product;
    static constexpr std::array parameter_names{
        std::string_view{"product.strike"},
    };

    __host__ __device__ static bool valid(const Product& product) {
        return ::isfinite(product.strike) && product.maturity_days > 0U;
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Product& product
    ) {
        return index == 0U ? product.strike : ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Product& product,
        float value
    ) {
        if (index == 0U) product.strike = value;
    }
};

struct Dynamics {
    using ModelParameters = Model;
    using Prepared = Model;
    struct State { float value; };
    struct RandomContext {
        std::size_t path;
        __device__ RandomContext(philox::PhiloxKey, std::size_t value)
            : path(value) {}
    };

    static constexpr bool kExactTerminal = true;
    static constexpr bool kDrawRequiresCentralPrepared = false;

    __device__ static Prepared prepare(Model model, float) { return model; }
    __device__ static State initial(Prepared model) { return {model.spot}; }
    __device__ static float draw(RandomContext& random) {
        return static_cast<float>(1U + random.path % 3U);
    }
    __device__ static void transition(
        Prepared model,
        float innovation,
        const float*,
        State& state
    ) {
        atomicAdd(&work_counts[0], 1U);
        if (model.coefficient == 0.5f
            && model.boundary_coefficient == 0.0f) {
            atomicAdd(&work_counts[1], 1U);
        }
        state.value += (model.coefficient + model.boundary_coefficient)
            * innovation;
    }
    __device__ static float spot(State state) { return state.value; }
};

struct Payoff {
    struct PreparedProduct { float strike; };

    __device__ static PreparedProduct prepare_product(
        Model,
        Product product,
        equity::ProductPreparationContext
    ) {
        return {product.strike};
    }
    __device__ static int make_handler(PreparedProduct) { return 0; }

    template<typename Observation>
    __device__ static float finalize(
        PreparedProduct product,
        const typename Observation::State& state,
        int
    ) {
        atomicAdd(&work_counts[2], 1U);
        return Observation::spot(state) - product.strike;
    }

    template<typename Observation>
    __device__ static float centered_difference(
        PreparedProduct first,
        PreparedProduct second,
        const typename Observation::State& first_state,
        const typename Observation::State& second_state,
        float width
    ) {
        return (finalize<Observation>(second, second_state, 0)
                - finalize<Observation>(first, first_state, 0))
            / width;
    }
};

using Plan = epg::TerminalDevicePreparedPlan<
    ModelPreparation,
    ProductPreparation
>;

void run(std::size_t block_count) {
    constexpr std::size_t rows = 2U;
    constexpr std::size_t sensitivity_count = 4U;
    constexpr std::size_t paths = 65U;
    const std::vector<Model> models(rows, {1.0f, 0.5f, 0.0f});
    const std::vector<Product> products(rows, {0.0f, 1U});
    const pg::PriceGradientConfiguration configuration{{
        {"model.spot", {.125, pg::BumpScale::relative}},
        {"model.coefficient", {.125, pg::BumpScale::absolute}},
        {"product.strike", {.125, pg::BumpScale::absolute}},
        {"model.boundary_coefficient", {
            .125,
            pg::BumpScale::absolute,
            pg::BoundaryRule::central_then_one_sided_order2,
        }},
    }};
    const auto plan = epg::prepare_terminal_device_sensitivities<Plan>(
        std::span<const Model>(models),
        std::span<const Product>(products),
        PriceConstruction::Aligned,
        {1.0f, 1U},
        configuration,
        {pg::SensitivityOrders::first}
    );

    DeviceArray<Model> device_models(plan.models);
    DeviceArray<Product> device_products(plan.products);
    DeviceArray<Plan::SensitivitySpec> sensitivities(plan.sensitivities);
    DeviceArray<pg::SensitivityStencil<3U>> stencils(
        rows * sensitivity_count
    );
    DeviceArray<pg::device_preparation::Error> error(1U);
    DeviceArray<float> values(2U * rows * (1U + sensitivity_count));
    const Plan::DeviceInputs inputs{
        device_models.data,
        device_models.count,
        device_products.data,
        device_products.count,
        sensitivities.data,
        sensitivities.count,
    };
    const Plan::StencilOutputs stencil_outputs{
        stencils.data,
        stencils.count,
        error.data,
    };
    const pg::Outputs outputs{
        values.data,
        values.data + rows,
        values.data + 2U * rows,
        values.data + 2U * rows + rows * sensitivity_count,
        rows,
        rows * sensitivity_count,
    };
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        paths,
        64U,
        block_count,
        719U,
        1U,
    };
    const unsigned int zero[3]{};
    check_cuda(
        cudaMemcpyToSymbol(work_counts, zero, sizeof(zero)),
        "reset compact work counters"
    );
    epg::launch_terminal_first_sensitivities<Dynamics, Payoff>(
        plan,
        inputs,
        stencil_outputs,
        launch,
        outputs,
        [] {},
        "test.price_gradients.central_work",
        "compact/B=1"
    );
    check_cuda(cudaDeviceSynchronize(), "compact central-work kernel");

    unsigned int counts[3]{};
    check_cuda(
        cudaMemcpyFromSymbol(counts, work_counts, sizeof(counts)),
        "read compact work counters"
    );
    require(
        counts[0] == 7U * paths * rows,
        "Compact kernel performed redundant or missing transitions."
    );
    require(
        counts[1] == 3U * paths * rows,
        "Compact kernel performed redundant or missing central transitions."
    );
    require(
        counts[2] == 10U * paths * rows,
        "Compact kernel performed redundant or missing payoff evaluations."
    );

    const auto host = values.read();
    double innovation_sum = 0.0;
    for (std::size_t path = 0U; path < paths; ++path) {
        innovation_sum += 1U + path % 3U;
    }
    const float mean_innovation = static_cast<float>(innovation_sum / paths);
    for (std::size_t row = 0U; row < rows; ++row) {
        const auto gradient = 2U * rows + row * sensitivity_count;
        require(
            std::abs(host[row] - (1.0f + 0.5f * mean_innovation)) < 1.e-6f,
            "Compact central price is incorrect."
        );
        require(
            std::abs(
                host[gradient] - (1.0f + 0.5f * mean_innovation)
            ) < 1.e-6f,
            "Multiplicative spot sensitivity is incorrect."
        );
        require(
            std::abs(host[gradient + 1U] - mean_innovation) < 1.e-6f,
            "Dynamics sensitivity is incorrect."
        );
        require(
            std::abs(host[gradient + 2U] + 1.0f) < 1.e-6f,
            "Product sensitivity is incorrect."
        );
        require(
            std::abs(host[gradient + 3U] - mean_innovation) < 1.e-6f,
            "One-sided dynamics sensitivity is incorrect."
        );
    }
    require(
        error.read()[0U].code == pg::device_preparation::valid,
        "Compact device preparation reported an error."
    );
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess
            || device_count == 0) {
            return 77;
        }
        run(4U);
        run(8U);
        std::cout
            << "Compact central-work ownership and grid-stride reuse passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
