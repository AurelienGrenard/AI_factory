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
#include "tests/price_gradients/mixed_node_graph_cuda_test_support.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
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
struct FrozenPolicyOracleState {
    float spot;
    float factor;
};

struct FrozenPolicyOracleRow {
    float strike;
    float central_inverse_strike;
    float central_inverse_factor_scale;
    std::uint32_t regression_count;
};

struct FrozenPolicyOracleDynamics {
    using State = FrozenPolicyOracleState;
};

struct FrozenPolicyOracleRegressor {
    static constexpr std::size_t kBasisSize = 3U;

    struct Input {
        float normalized_spot;
        float normalized_factor;
    };

    struct Features {
        float values[kBasisSize];
    };

    __device__ __forceinline__ static Features evaluate(const Input& input) {
        return {{
            1.0f,
            input.normalized_spot,
            input.normalized_factor,
        }};
    }

    __device__ __forceinline__ static double predict(
        const Features& features,
        const double* coefficients
    ) {
        double value = 0.0;
        #pragma unroll
        for (std::size_t index = 0U; index < kBasisSize; ++index) {
            value += coefficients[index]
                * static_cast<double>(features.values[index]);
        }
        return value;
    }
};

struct FrozenPolicyOraclePricingPolicy {
    using Dynamics = FrozenPolicyOracleDynamics;
    using PreparedRow = FrozenPolicyOracleRow;

    __device__ __forceinline__ static float replay_immediate_value(
        const PreparedRow&,
        const PreparedRow& bumped,
        const Dynamics::State& state,
        std::uint32_t
    ) {
        return fmaxf(bumped.strike - state.spot, 0.0f);
    }

    __device__ __forceinline__ static FrozenPolicyOracleRegressor::Input
    replay_regression_input(
        const PreparedRow& central,
        const Dynamics::State& state
    ) {
        return {
            state.spot * central.central_inverse_strike,
            state.factor * central.central_inverse_factor_scale,
        };
    }

    __device__ __forceinline__ static bool regression_candidate(
        float immediate
    ) {
        return immediate > 0.0f;
    }
};

using FrozenPolicyOracleSnapshot =
    longstaff_schwartz::price_gradients::FrozenRegressionSnapshot<
        FrozenPolicyOracleRegressor
    >;

__global__ void frozen_policy_oracle_kernel(
    FrozenPolicyOracleRow central,
    FrozenPolicyOracleRow bumped,
    const FrozenPolicyOracleState* states,
    const FrozenPolicyOracleSnapshot* snapshots,
    std::size_t path_count,
    std::uint32_t* exercises
) {
    const std::size_t path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= path_count) return;

    longstaff_schwartz::price_gradients::FrozenRegressionExerciseHandler<
        FrozenPolicyOraclePricingPolicy,
        FrozenPolicyOracleRegressor
    > handler{
        central,
        bumped,
        snapshots,
        central.regression_count,
    };
    const std::size_t state_offset =
        path * (static_cast<std::size_t>(central.regression_count) + 1U);
    for (std::uint32_t observation = 0U;
         observation <= central.regression_count;
         ++observation) {
        if (!handler.on_observation(
                observation, states[state_offset + observation]
            )) {
            break;
        }
    }
    exercises[path] = handler.exercise;
}

std::vector<std::uint32_t> frozen_policy_cpu_oracle(
    const FrozenPolicyOracleRow& central,
    const FrozenPolicyOracleRow& bumped,
    const std::vector<FrozenPolicyOracleState>& states,
    const std::vector<FrozenPolicyOracleSnapshot>& snapshots,
    std::size_t path_count
) {
    std::vector<std::uint32_t> exercises(
        path_count, central.regression_count
    );
    const std::size_t states_per_path =
        static_cast<std::size_t>(central.regression_count) + 1U;
    for (std::size_t path = 0U; path < path_count; ++path) {
        for (std::uint32_t observation = 0U;
             observation < central.regression_count;
             ++observation) {
            const auto& snapshot = snapshots[observation];
            if (snapshot.status
                != longstaff_schwartz::RegressionStatus::success) {
                continue;
            }
            const auto& state =
                states[path * states_per_path + observation];
            const float immediate =
                std::max(bumped.strike - state.spot, 0.0f);
            if (immediate <= 0.0f) continue;
            const double features[] = {
                1.0,
                static_cast<double>(
                    state.spot * central.central_inverse_strike
                ),
                static_cast<double>(
                    state.factor * central.central_inverse_factor_scale
                ),
            };
            double continuation = 0.0;
            for (std::size_t basis = 0U;
                 basis < FrozenPolicyOracleRegressor::kBasisSize;
                 ++basis) {
                continuation += snapshot.coefficients[basis]
                    * features[basis];
            }
            if (static_cast<double>(immediate) > continuation) {
                exercises[path] = observation;
                break;
            }
        }
    }
    return exercises;
}

void check_frozen_policy_small_path_oracle() {
    constexpr std::size_t path_count = 5U;
    const FrozenPolicyOracleRow central{1.0f, 1.0f, 0.5f, 3U};
    const FrozenPolicyOracleRow bumped{1.05f, 1.0f, 0.5f, 3U};
    const std::vector<FrozenPolicyOracleSnapshot> snapshots{
        {{0.0, 0.067, 0.0},
         longstaff_schwartz::RegressionStatus::success},
        {{0.0, 0.0, 0.0},
         longstaff_schwartz::RegressionStatus::no_candidates},
        {{0.03, 0.0, 0.0},
         longstaff_schwartz::RegressionStatus::success},
    };
    const std::vector<FrozenPolicyOracleState> states{
        {0.970f, 0.0f}, {0.930f, 0.0f},
        {0.880f, 0.0f}, {0.850f, 0.0f},
        {0.985f, 0.0f}, {0.920f, 0.0f},
        {0.900f, 0.0f}, {0.850f, 0.0f},
        {1.060f, 0.0f}, {0.900f, 0.0f},
        {1.040f, 0.0f}, {0.800f, 0.0f},
        {0.950f, 0.0f}, {0.940f, 0.0f},
        {0.930f, 0.0f}, {0.900f, 0.0f},
        {1.200f, 0.0f}, {0.920f, 0.0f},
        {0.980f, 0.0f}, {0.900f, 0.0f},
    };

    const auto expected = frozen_policy_cpu_oracle(
        central, bumped, states, snapshots, path_count
    );
    const auto central_expected = frozen_policy_cpu_oracle(
        central, central, states, snapshots, path_count
    );
    price_gradient_test::require(
        expected == std::vector<std::uint32_t>({0U, 2U, 3U, 0U, 2U}),
        "Frozen-policy CPU oracle does not exercise the intended cases"
    );
    price_gradient_test::require(
        central_expected[0U] != expected[0U],
        "Strike bump did not change the oracle exercise decision"
    );

    price_gradient_test::DeviceArray<FrozenPolicyOracleState>
        device_states(states);
    price_gradient_test::DeviceArray<FrozenPolicyOracleSnapshot>
        device_snapshots(snapshots);
    price_gradient_test::DeviceArray<std::uint32_t>
        device_exercises(path_count);
    frozen_policy_oracle_kernel<<<1U, 32U>>>(
        central,
        bumped,
        device_states.data,
        device_snapshots.data,
        path_count,
        device_exercises.data
    );
    check_cuda(
        cudaGetLastError(), "launch frozen-policy small-path oracle"
    );
    check_cuda(
        cudaDeviceSynchronize(), "synchronize frozen-policy small-path oracle"
    );
    price_gradient_test::require(
        device_exercises.read() == expected,
        "CUDA frozen-policy replay differs from the independent CPU oracle"
    );
}


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

void check_black_scholes_frozen_regression_policy() {
    using Contract = ModelContract<BlackScholesTag>;
    const auto models = Contract::models();
    const std::vector<product::AmericanOptionParameters> products{
        {1.0f, 63U, 7U}
    };
    const pg::TimeConfiguration time{1.0f / 504.0f, 2U};
    const auto plan = Contract::prepare(
        models,
        products,
        time,
        Contract::sensitivities(),
        pg::SensitivityOrders::first_and_second
    );
    const auto frozen_time = american_test::execute<
        pg::SensitivityOrders::first_and_second
    >(
        plan, test_paths, 128U, 4U, 91871U,
        Contract::diagonal_launcher()
    );
    // Independent QuantLib 1.43 CRR Bermudan reference: Business252 with a
    // NullCalendar, exercise at days 9, 18, ..., 63, and 4032 tree steps.
    // This checks the central LSM price level, beyond launcher parity.
    constexpr double bermudan_reference = 0.03752383405936103;
    const double price_tolerance =
        6.0 * frozen_time.price_errors[0U] + 0.001;
    if (std::abs(frozen_time.prices[0U] - bermudan_reference)
        >= price_tolerance) {
        std::cerr << "Black-Scholes Bermudan price mismatch: observed="
                  << frozen_time.prices[0U]
                  << " reference=" << bermudan_reference
                  << " standard_error=" << frozen_time.price_errors[0U]
                  << '\n';
        throw std::runtime_error(
            "American LSM price misses independent Bermudan reference."
        );
    }
    const auto frozen_policy = american_test::execute<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        test_paths,
        128U,
        4U,
        91871U,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return bs::
                launch_black_scholes_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    host, inputs, stencils, launch, outputs,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        }
    );
    american_test::require_finite(
        frozen_policy, "Black-Scholes frozen-policy"
    );
    const auto graph_policy = american_test::execute<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        test_paths,
        128U,
        4U,
        91871U,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            const auto bytes = bs::
                black_scholes_american_option_node_graph_workspace_bytes_with_replay<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    host,
                    launch,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
            price_gradient_test::DeviceArray<std::uint8_t> workspace(bytes);
            return bs::
                launch_black_scholes_american_option_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    host, inputs, stencils, launch, outputs,
                    workspace.data, workspace.count,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        }
    );
    price_gradient_test::require(
        frozen_time.prices == frozen_policy.prices,
        "American replay strategy changed central prices"
    );
    price_gradient_test::require(
        frozen_policy.gradients == graph_policy.gradients
            && frozen_policy.gradient_errors == graph_policy.gradient_errors
            && frozen_policy.diagonal_hessians
                == graph_policy.diagonal_hessians
            && frozen_policy.diagonal_hessian_errors
                == graph_policy.diagonal_hessian_errors,
        "American frozen-policy mono and node graph differ"
    );

    const auto mixed_plan = bs::
        prepare_black_scholes_american_option_sensitivities(
            models,
            products,
            PriceConstruction::Aligned,
            time,
            Contract::sensitivities(),
            pg::SensitivityRequest::full_hessian()
        );
    price_gradient_test::require_mixed_node_graph_parity(
        mixed_plan,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return bs::
                launch_black_scholes_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    host, inputs, stencils, launch, outputs,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        },
        [](const auto& host, const auto& launch) {
            return bs::
                black_scholes_american_option_mixed_node_graph_workspace_bytes_with_replay<
                    OptionSide::put
                >(
                    host,
                    launch,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        },
        [](const auto& host, auto inputs, auto stencils,
           auto mixed_stencils, const auto& launch,
           auto outputs, auto mixed_outputs,
           void* workspace, std::size_t workspace_bytes) {
            return bs::
                launch_black_scholes_american_option_mixed_node_graph_sensitivities_with_replay_cuda<
                    OptionSide::put
                >(
                    host, inputs, stencils, mixed_stencils,
                    launch, outputs, mixed_outputs,
                    workspace, workspace_bytes,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        },
        257U,
        91871U,
        "Black-Scholes American frozen-policy mixed node graph"
    );
}


void check_black_scholes_frozen_policy_initial_decision() {
    const std::vector<bs::ModelParameters> models{
        {1.0f, 0.03f, 0.0f, 0.01f}
    };
    const std::vector<product::AmericanOptionParameters> products{
        {1.0f, 2U, 1U}
    };
    const pg::TimeConfiguration time{1.0f / 252.0f, 1U};
    const pg::PriceGradientConfiguration selection{{
        {"product.strike", {0.10f}}
    }};
    const auto plan = bs::prepare_black_scholes_american_option_sensitivities(
        models,
        products,
        PriceConstruction::Aligned,
        time,
        selection,
        {pg::SensitivityOrders::first_and_second}
    );
    const auto frozen_policy = american_test::execute<
        pg::SensitivityOrders::first_and_second
    >(
        plan,
        test_paths,
        128U,
        4U,
        42017U,
        [](const auto& host, auto inputs, auto stencils,
           const auto& launch, auto outputs) {
            return bs::
                launch_black_scholes_american_option_diagonal_sensitivities_with_replay_cuda<
                    OptionSide::put,
                    pg::SensitivityOrders::first_and_second
                >(
                    host, inputs, stencils, launch, outputs,
                    longstaff_schwartz::price_gradients::
                        ExerciseReplayStrategy::frozen_regression_policy
                );
        }
    );
    american_test::require_finite(
        frozen_policy, "Black-Scholes frozen-policy initial decision"
    );
    price_gradient_test::require(
        std::abs(frozen_policy.gradients[0U] - 0.5f) < 1.0e-5f,
        "Frozen policy did not exercise the upper strike node at t0"
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
        check_frozen_policy_small_path_oracle();
        verify<BlackScholesTag>();
        check_black_scholes_frozen_regression_policy();
        check_black_scholes_frozen_policy_initial_decision();
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
