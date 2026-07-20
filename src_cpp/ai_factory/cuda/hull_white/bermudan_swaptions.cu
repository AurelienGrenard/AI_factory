#include "ai_factory/cuda/hull_white/bermudan_swaptions.cuh"

#include "ai_factory/common/fixed_income/hull_white.hpp"
#include "ai_factory/cuda/common/bermudan_lsm.cuh"
#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

namespace ai_factory::cuda {
namespace {

constexpr std::size_t kMaxExercises = bermudan_lsm::kMaxExercises;
constexpr std::size_t kMaxPayments = 20U;
constexpr std::size_t kRowsPerBatch = 128U;
constexpr double kBasisRateScale = 0.04;
struct HullWhiteBermudanWorkspaceTag {};
struct ExerciseCoefficients {
    double decay;
    double state_scale;
    double integral_state_loading;
    double integral_loading;
    double residual_scale;
    double deterministic_integral;
};

__global__ void precompute_bonds_kernel(
    const HullWhiteBermudanSwaptionRow* rows,
    std::size_t row_count,
    int* exercise_counts,
    double* bond_a,
    double* bond_b,
    ExerciseCoefficients* exercise_coefficients
) {
    const std::size_t row_index = blockIdx.x;
    if (row_index >= row_count) return;
    const auto row = rows[row_index];
    if (threadIdx.x == 0) exercise_counts[row_index] = row.product.exercise_count;
    if (threadIdx.x < row.product.exercise_count) {
        const std::size_t exercise = threadIdx.x;
        const double time = row.product.first_exercise
            + exercise * row.product.exercise_period;
        const double previous_time = exercise == 0U
            ? 0.0
            : row.product.first_exercise
                + (exercise - 1U) * row.product.exercise_period;
        const double interval = time - previous_time;
        const double state_variance = fixed_income::hull_white_state_variance(
            row.mean_reversion, row.volatility, interval
        );
        const double integral_variance = fixed_income::hull_white_integral_variance(
            row.mean_reversion, row.volatility, interval
        );
        const double covariance = fixed_income::hull_white_state_integral_covariance(
            row.mean_reversion, row.volatility, interval
        );
        auto& coefficients = exercise_coefficients[
            row_index * kMaxExercises + exercise
        ];
        coefficients.decay = exp(-row.mean_reversion * interval);
        coefficients.state_scale = sqrt(state_variance);
        coefficients.integral_state_loading = fixed_income::hull_white_b(
            row.mean_reversion, interval
        );
        coefficients.integral_loading = covariance / coefficients.state_scale;
        coefficients.residual_scale = sqrt(fmax(
            integral_variance - coefficients.integral_loading
                * coefficients.integral_loading,
            0.0
        ));
        coefficients.deterministic_integral =
            fixed_income::hull_white_deterministic_integral(
                time, row.mean_reversion, row.volatility,
                row.beta0, row.beta1, row.beta2, row.tau
            );
    }
    for (std::size_t flat = threadIdx.x; flat < kMaxExercises * kMaxPayments;
         flat += blockDim.x) {
        const std::size_t exercise = flat / kMaxPayments;
        const std::size_t payment = flat % kMaxPayments;
        const std::size_t output = (row_index * kMaxExercises + exercise)
            * kMaxPayments + payment;
        if (exercise < static_cast<std::size_t>(row.product.exercise_count)
            && payment >= exercise
            && payment < static_cast<std::size_t>(row.product.payment_count)) {
            const double time = row.product.first_exercise
                + exercise * row.product.exercise_period;
            const double maturity = row.product.first_exercise
                + (payment + 1U) * row.product.accrual_period;
            bond_a[output] = fixed_income::hull_white_bond_a(
                time, maturity, row.mean_reversion, row.volatility,
                row.beta0, row.beta1, row.beta2, row.tau
            );
            bond_b[output] = fixed_income::hull_white_b(
                row.mean_reversion, maturity - time
            );
        } else {
            bond_a[output] = 0.0;
            bond_b[output] = 0.0;
        }
    }
}

__global__ void simulate_exercise_values_kernel(
    const HullWhiteBermudanSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    const double* bond_a,
    const double* bond_b,
    const ExerciseCoefficients* exercise_coefficients,
    double* immediate,
    double* basis_states,
    double* discounts
) {
    const std::size_t row_index = blockIdx.x;
    if (row_index >= row_count) return;
    const auto row = rows[row_index];
    const auto product = row.product;
    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        rng::NormalSequence normals(
            row.seed, 0U, path * 2U * static_cast<std::size_t>(product.exercise_count)
        );
        double state = 0.0;
        double stochastic_integral = 0.0;
        for (int exercise = 0; exercise < product.exercise_count; ++exercise) {
            const auto coefficients = exercise_coefficients[
                row_index * kMaxExercises + exercise
            ];
            const double first_normal = normals.next();
            const double second_normal = normals.next();
            const double previous_state = state;
            state = coefficients.decay * state
                + coefficients.state_scale * first_normal;
            stochastic_integral += coefficients.integral_state_loading
                * previous_state + coefficients.integral_loading * first_normal
                + coefficients.residual_scale * second_normal;

            const std::size_t coefficient_offset =
                (row_index * kMaxExercises + exercise) * kMaxPayments;
            double annuity = 0.0;
            double end_bond = 1.0;
            for (int payment = exercise; payment < product.payment_count; ++payment) {
                const double bond = bond_a[coefficient_offset + payment]
                    * exp(-bond_b[coefficient_offset + payment] * state);
                annuity += product.accrual_period * bond;
                end_bond = bond;
            }
            const double signed_swap = static_cast<double>(product.direction)
                * (1.0 - end_bond - product.fixed_rate * annuity);
            const std::size_t output =
                (row_index * kMaxExercises + exercise) * num_paths + path;
            immediate[output] = product.notional * fmax(signed_swap, 0.0);
            basis_states[output] = ((1.0 - end_bond) / annuity) / kBasisRateScale;
            discounts[output] = exp(
                -coefficients.deterministic_integral - stochastic_integral
            );
        }
    }
}

}  // namespace

void price_hull_white_bermudan_swaption_cuda(
    const HullWhiteBermudanSwaptionRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    if (row_count == 0U) return;
    auto& workspace = detail::reusable_cuda_workspace<
        HullWhiteBermudanWorkspaceTag, 12U
    >();
    const std::size_t capacity = row_count < kRowsPerBatch ? row_count : kRowsPerBatch;
    auto* rows = workspace.buffer<HullWhiteBermudanSwaptionRow>(0, capacity, "bermudan rows");
    auto* counts = workspace.buffer<int>(1, capacity, "bermudan exercise counts");
    auto* bond_a = workspace.buffer<double>(2, capacity * kMaxExercises * kMaxPayments, "bermudan bond a");
    auto* bond_b = workspace.buffer<double>(3, capacity * kMaxExercises * kMaxPayments, "bermudan bond b");
    auto* exercise_coefficients = workspace.buffer<ExerciseCoefficients>(
        11, capacity * kMaxExercises, "bermudan exercise coefficients"
    );
    const std::size_t state_capacity = capacity * kMaxExercises * num_paths;
    auto* immediate = workspace.buffer<double>(4, state_capacity, "bermudan immediate");
    auto* basis = workspace.buffer<double>(5, state_capacity, "bermudan basis");
    auto* discounts = workspace.buffer<double>(6, state_capacity, "bermudan discounts");
    auto* cashflows = workspace.buffer<double>(7, capacity * num_paths, "bermudan cashflows");
    auto* stats = workspace.buffer<double>(8, capacity * bermudan_lsm::kStatCount, "bermudan stats");
    auto* coefficients = workspace.buffer<double>(9, capacity * (bermudan_lsm::kBasisSize + 1U), "bermudan coefficients");
    auto* outputs = workspace.buffer<MonteCarloOutput>(10, capacity, "bermudan outputs");
    float kernel_ms = 0.0F;
    for (std::size_t offset = 0; offset < row_count; offset += capacity) {
        const std::size_t remaining = row_count - offset;
        const std::size_t batch = remaining < capacity ? remaining : capacity;
        detail::check_cuda(cudaMemcpy(rows, host_rows + offset, batch * sizeof(*rows), cudaMemcpyHostToDevice), "copy bermudan rows");
        detail::check_cuda(cudaEventRecord(workspace.start_event()), "bermudan event start");
        precompute_bonds_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(
            rows, batch, counts, bond_a, bond_b, exercise_coefficients
        );
        simulate_exercise_values_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(
            rows, batch, num_paths, bond_a, bond_b, exercise_coefficients,
            immediate, basis, discounts
        );
        bermudan_lsm::initialize_cashflows_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(
            counts, immediate, discounts, batch, num_paths, cashflows
        );
        const auto regression_shared = bermudan_lsm::kStatCount
            * detail::kThreadsPerBlock * sizeof(double);
        for (std::size_t exercise = kMaxExercises - 1U; exercise-- > 0U;) {
            bermudan_lsm::regression_stats_kernel<<<
                static_cast<unsigned int>(batch), detail::kThreadsPerBlock, regression_shared
            >>>(counts, immediate, basis, discounts, cashflows, batch, num_paths, exercise, stats);
            bermudan_lsm::solve_kernel<<<static_cast<unsigned int>(batch), 1>>>(stats, batch, coefficients);
            bermudan_lsm::apply_exercise_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(
                counts, immediate, basis, discounts, coefficients, batch, num_paths, exercise, cashflows
            );
        }
        bermudan_lsm::finalize_kernel<<<
            static_cast<unsigned int>(batch), detail::kThreadsPerBlock,
            2U * detail::kThreadsPerBlock * sizeof(double)
        >>>(cashflows, batch, num_paths, outputs);
        detail::check_cuda(cudaGetLastError(), "Hull-White Bermudan kernels");
        detail::check_cuda(cudaEventRecord(workspace.stop_event()), "bermudan event stop");
        detail::check_cuda(cudaEventSynchronize(workspace.stop_event()), "bermudan event sync");
        float elapsed = 0.0F;
        detail::check_cuda(cudaEventElapsedTime(&elapsed, workspace.start_event(), workspace.stop_event()), "bermudan timing");
        kernel_ms += elapsed;
        detail::check_cuda(cudaMemcpy(host_outputs + offset, outputs, batch * sizeof(*outputs), cudaMemcpyDeviceToHost), "copy bermudan outputs");
    }
    if (timing != nullptr) {
        timing->simulation_ms = kernel_ms;
        timing->total_ms = kernel_ms;
    }
}

}  // namespace ai_factory::cuda
