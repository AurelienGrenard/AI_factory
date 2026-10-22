// CEV frozen-date replay oracle (including absorption) and fatal delta propagation.
#include "tests/price_delta/cuda_test_support.cuh"
#include "common/equity/price_delta/coupled_spot_paths.cuh"
#include "model/equity/markovian/cev/price_delta_dynamics_impl.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_delta_policy.cuh"
#include <array>
#include <bit>
#include <cmath>
#include <iostream>
#include <vector>

using namespace ai_factory::workbench;
namespace cev = model::equity::cev;
namespace pd = equity::price_delta;
namespace lsm = longstaff_schwartz;
using price_delta_test::DeviceArray;
using price_delta_test::require;
using Schedule = simulation::FixedStepMaturityAlignedExerciseSchedule<cev::DynamicsPolicy>;
using Replay = pd::CoupledFrozenExercise<
    simulation::FixedStepMaturityAlignedExerciseSchedule<cev::PriceDeltaDynamics>,
    pd::CoupledSpotPaths<cev::PriceDeltaDynamics>>;
using Policy = product::AmericanOptionPriceDeltaPolicy<Schedule, OptionSide::put,
    product::SpotLogMoneynessContinuationState<cev::DynamicsPolicy>, Replay>;

// Independent orchestration: directly advance the original single-spot dynamics
// through the stub and selected exercise intervals, without the replay handler.
__device__ float original_spot(cev::ModelParameters model, philox::PhiloxKey key,
                              std::size_t path, std::uint32_t exercise) {
    const auto prepared = cev::DynamicsPolicy::prepare_dynamics(model, 1.0f / 504);
    auto state = cev::DynamicsPolicy::initial_state(prepared);
    cev::DynamicsPolicy::RandomContext random(key, path);
    cev::DynamicsPolicy::advance(prepared, 12U, random, state);  // 20 - 2*7 days
    for (std::uint32_t index = 0; index < exercise; ++index)
        cev::DynamicsPolicy::advance(prepared, 14U, random, state);
    return cev::DynamicsPolicy::spot(state);
}

__global__ void check_replay(cev::ModelParameters model, float* spots,
                            pd::FrozenExercise* exercises, int* failures) {
    constexpr std::size_t paths = 256;
    const std::size_t path = threadIdx.x;
    const auto key = philox::make_key(79123);
    Policy::StateView states{{spots}, exercises, nullptr};
    auto row = Policy::prepare_row(model, {1.0f, 20U, 7U}, key, 0, 0, paths,
                                   {.01f}, {1.0f / 504, 2U});
    row.exercises = exercises;
    Policy::simulate_path(row, path, paths, states);
    const auto selected = static_cast<std::uint32_t>(path % 3U);
    if (selected < row.regression_count) {
        Policy::record_exercise(row, states, selected * paths + path, path,
                                 row.regression_count - 1U - selected);
    }
    const auto trace = exercises[path];
    const float central = original_spot(model, key, path, selected);
    auto lower_model = model, upper_model = model;
    lower_model.spot = row.bump.lower;
    upper_model.spot = row.bump.upper;
    float lower = fmaxf(1.0f - original_spot(lower_model, key, path, selected), 0.0f);
    float upper = fmaxf(1.0f - original_spot(upper_model, key, path, selected), 0.0f);
    for (std::uint32_t date = 0; date < selected; ++date) {
        lower = row.exercise_discount * lower;
        upper = row.exercise_discount * upper;
    }
    lower = row.initial_discount * lower;
    upper = row.initial_discount * upper;
    const float expected = (upper - lower) / row.bump.width;
    failures[path] = trace.observation != selected || __float_as_uint(trace.spot) != __float_as_uint(central)
        || __float_as_uint(Policy::path_delta(row, path)) != __float_as_uint(expected);
}

void check_invalid_results() {
    DeviceArray<Policy::PreparedRow> device_rows(2);
    DeviceArray<Policy::InitialDecision> decisions(2);
    DeviceArray<lsm::RegressionDiagnostics> device_diagnostics(2);
    DeviceArray<double> partials(4);
    DeviceArray<float> deltas(2), errors(2);
    std::array<Policy::PreparedRow, 2> rows{};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        rows[i].result_index = i;
        rows[i].initial_exercise = decisions.data + i;
    }
    const std::array<Policy::InitialDecision, 2> initial{
        Policy::InitialDecision::continuation, Policy::InitialDecision::invalid};
    std::array<lsm::RegressionDiagnostics, 2> diagnostics{};
    diagnostics[0].fatal_failure_count = 1;
    check_cuda(cudaMemcpy(device_rows.data, rows.data(), sizeof(rows), cudaMemcpyHostToDevice), "rows");
    check_cuda(cudaMemcpy(decisions.data, initial.data(), sizeof(initial), cudaMemcpyHostToDevice), "decisions");
    check_cuda(cudaMemcpy(device_diagnostics.data, diagnostics.data(), sizeof(diagnostics), cudaMemcpyHostToDevice), "diagnostics");
    lsm::finalize_frozen_date_deltas_kernel<Policy><<<2, 128, 64>>>(
        device_rows.data, 256, 1, device_diagnostics.data, partials.data, deltas.data, errors.data);
    check_cuda(cudaGetLastError(), "invalid delta finalization");
    std::array<float, 2> values{}, standard_errors{};
    check_cuda(cudaMemcpy(values.data(), deltas.data, sizeof(values), cudaMemcpyDeviceToHost), "invalid deltas");
    check_cuda(cudaMemcpy(standard_errors.data(), errors.data, sizeof(standard_errors), cudaMemcpyDeviceToHost), "invalid errors");
    for (std::size_t i = 0; i < 2; ++i)
        require(std::isnan(values[i]) && std::isnan(standard_errors[i]), "Invalid central result did not invalidate delta");
}

void check_workspace_budget() {
    lsm::WorkspaceDescriptor descriptor{sizeof(Policy::PreparedRow), alignof(Policy::PreparedRow),
        Policy::state_field_descriptors(), 6U, 28U};
    const auto baseline = lsm::make_workspace_layout(descriptor, 1, 512, 256, 2, "trace test");
    descriptor.path_fields = Policy::path_field_descriptors();
    descriptor.row_fields = Policy::row_field_descriptors();
    const auto traced = lsm::make_workspace_layout(descriptor, 1, 512, 256, 2, "trace test");
    require(traced.total_bytes == baseline.total_bytes + 256 * sizeof(pd::FrozenExercise)
                                    + sizeof(Policy::InitialDecision), "Incorrect trace allocation");
    const auto plan = lsm::plan_batches({{2, 512}, {2, 512}}, descriptor,
                                       256, 2, traced.total_bytes, "trace test");
    require(plan.batches.size() == 2 && plan.maximum_prices_per_batch == 1,
            "Trace storage was omitted from the batch planner");
    require(plan.maximum_workspace_bytes == traced.total_bytes, "Incorrect maximum workspace");
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        check_workspace_budget();
        DeviceArray<float> spots(512);
        DeviceArray<pd::FrozenExercise> exercises(256);
        DeviceArray<int> failures(256);
        for (const auto& model : std::array<cev::ModelParameters, 2>{{
                {1.2f, .03f, .01f, .3f, .6f}, {.05f, .03f, .01f, 1.8f, .4f}}}) {
            check_replay<<<1, 256>>>(model, spots.data, exercises.data, failures.data);
            check_cuda(cudaGetLastError(), "check frozen CEV replay");
            std::vector<int> results(256);
            check_cuda(cudaMemcpy(results.data(), failures.data, 256 * sizeof(int), cudaMemcpyDeviceToHost), "replay results");
            for (int failure : results) require(failure == 0, "Frozen-date replay differs from original dynamics");
        }
        check_invalid_results();
        std::cout << "512 pathwise frozen-date checks and two invalid-result checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
