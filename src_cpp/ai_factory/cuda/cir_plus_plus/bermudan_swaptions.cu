#include "ai_factory/cuda/cir_plus_plus/bermudan_swaptions.cuh"

#include "ai_factory/common/fixed_income/cir_plus_plus.hpp"
#include "ai_factory/cuda/common/bermudan_lsm.cuh"
#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

namespace ai_factory::cuda {
namespace {

constexpr std::size_t kMaxExercises = bermudan_lsm::kMaxExercises;
constexpr std::size_t kMaxPayments = 20U;
constexpr std::size_t kRowsPerBatch = 128U;
constexpr double kBasisRateScale = 0.04;
constexpr double kQePsiCutoff = 1.5;
struct CirPlusPlusBermudanWorkspaceTag {};
struct StepCoefficients {
    std::size_t num_steps;
    double dt;
    double decay;
    double one_minus_decay;
    double volatility_squared;
};

__global__ void precompute_bonds_kernel(
    const CirPlusPlusBermudanSwaptionRow* rows, std::size_t row_count,
    int* exercise_counts, double* bond_a, double* bond_b
) {
    const std::size_t row_index = blockIdx.x;
    if (row_index >= row_count) return;
    const auto row = rows[row_index];
    if (threadIdx.x == 0) exercise_counts[row_index] = row.product.exercise_count;
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
            bond_a[output] = fixed_income::cir_plus_plus_bond_a(
                time, maturity, row.model.initial_factor,
                row.model.kappa, row.model.theta, row.model.volatility,
                row.curve.beta0, row.curve.beta1,
                row.curve.beta2, row.curve.tau
            );
            bond_b[output] = fixed_income::cir_bond_b(
                row.model.kappa, row.model.volatility, maturity - time
            );
        } else {
            bond_a[output] = 0.0;
            bond_b[output] = 0.0;
        }
    }
}

__global__ void precompute_steps_kernel(
    const CirPlusPlusBermudanSwaptionRow* rows,
    std::size_t row_count,
    double target_dt,
    StepCoefficients* coefficients
) {
    const std::size_t row_index = blockIdx.x;
    if (row_index >= row_count || threadIdx.x != 0) return;
    const auto row = rows[row_index];
    const double last_exercise = row.product.first_exercise
        + (row.product.exercise_count - 1) * row.product.exercise_period;
    const auto rounded_steps = llround(last_exercise / target_dt);
    auto& output = coefficients[row_index];
    output.num_steps = static_cast<std::size_t>(
        rounded_steps > 0LL ? rounded_steps : 1LL
    );
    output.dt = last_exercise / static_cast<double>(output.num_steps);
    output.decay = exp(-row.model.kappa * output.dt);
    output.one_minus_decay = 1.0 - output.decay;
    output.volatility_squared = row.model.volatility * row.model.volatility;
}

__global__ void simulate_exercise_values_kernel(
    const CirPlusPlusBermudanSwaptionRow* rows, std::size_t row_count,
    std::size_t num_paths,
    const StepCoefficients* step_coefficients,
    const double* bond_a, const double* bond_b,
    double* immediate, double* basis_states, double* discounts
) {
    const std::size_t row_index = blockIdx.x;
    if (row_index >= row_count) return;
    const auto row = rows[row_index];
    const auto model = row.model;
    const auto product = row.product;
    const auto coefficients = step_coefficients[row_index];
    for (std::size_t path = threadIdx.x; path < num_paths; path += blockDim.x) {
        rng::NormalSequence normals(row.seed, path, 0U);
        rng::UniformSequence uniforms(row.seed, path + num_paths, 0U);
        double rate = model.initial_factor;
        double integral = 0.0;
        int next_exercise = 0;
        std::size_t next_step = static_cast<std::size_t>(
            llround(product.first_exercise / coefficients.dt)
        );
        for (std::size_t step = 1U; step <= coefficients.num_steps; ++step) {
            const double previous_rate = rate;
            const double mean = model.theta
                + (rate - model.theta) * coefficients.decay;
            const double variance = rate * coefficients.volatility_squared
                * coefficients.decay * coefficients.one_minus_decay / model.kappa
                + model.theta * coefficients.volatility_squared
                    * coefficients.one_minus_decay * coefficients.one_minus_decay
                    / (2.0 * model.kappa);
            const double psi = variance / (mean * mean);
            const double normal = normals.next();
            const double uniform = uniforms.next();
            if (psi <= kQePsiCutoff) {
                const double inverse_psi = 1.0 / psi;
                const double b_squared = 2.0 * inverse_psi - 1.0
                    + sqrt(2.0 * inverse_psi)
                          * sqrt(fmax(2.0 * inverse_psi - 1.0, 0.0));
                const double shifted = sqrt(b_squared) + normal;
                rate = mean / (1.0 + b_squared) * shifted * shifted;
            } else {
                const double probability = (psi - 1.0) / (psi + 1.0);
                rate = uniform <= probability ? 0.0
                    : mean / (1.0 - probability)
                        * log((1.0 - probability) / (1.0 - uniform));
            }
            integral += 0.5 * (previous_rate + rate) * coefficients.dt;
            if (step == next_step && next_exercise < product.exercise_count) {
                const std::size_t coefficient_offset =
                    (row_index * kMaxExercises + next_exercise) * kMaxPayments;
                double annuity = 0.0;
                double end_bond = 1.0;
                for (int payment = next_exercise;
                     payment < product.payment_count; ++payment) {
                    const double bond = bond_a[coefficient_offset + payment]
                        * exp(-bond_b[coefficient_offset + payment] * rate);
                    annuity += product.accrual_period * bond;
                    end_bond = bond;
                }
                const double signed_swap = static_cast<double>(product.direction)
                    * (1.0 - end_bond - product.fixed_rate * annuity);
                const std::size_t output =
                    (row_index * kMaxExercises + next_exercise) * num_paths + path;
                immediate[output] = product.notional * fmax(signed_swap, 0.0);
                basis_states[output] = ((1.0 - end_bond) / annuity) / kBasisRateScale;
                const double exercise_time = product.first_exercise
                    + next_exercise * product.exercise_period;
                discounts[output] = fixed_income::cir_plus_plus_path_discount(
                    integral, exercise_time, model.initial_factor,
                    model.kappa, model.theta, model.volatility,
                    row.curve.beta0, row.curve.beta1,
                    row.curve.beta2, row.curve.tau
                );
                ++next_exercise;
                if (next_exercise < product.exercise_count) {
                    const double next_time = product.first_exercise
                        + next_exercise * product.exercise_period;
                    next_step = static_cast<std::size_t>(
                        llround(next_time / coefficients.dt)
                    );
                }
            }
        }
    }
}

}  // namespace

void price_cir_plus_plus_bermudan_swaption_cuda(
    const CirPlusPlusBermudanSwaptionRow* host_rows, std::size_t row_count,
    std::size_t num_paths, double target_dt,
    MonteCarloOutput* host_outputs, CudaTiming* timing
) {
    if (row_count == 0U) return;
    auto& workspace = detail::reusable_cuda_workspace<CirPlusPlusBermudanWorkspaceTag, 12U>();
    const std::size_t capacity = row_count < kRowsPerBatch ? row_count : kRowsPerBatch;
    auto* rows = workspace.buffer<CirPlusPlusBermudanSwaptionRow>(0, capacity, "CIR++ Bermudan rows");
    auto* counts = workspace.buffer<int>(1, capacity, "CIR++ Bermudan counts");
    auto* bond_a = workspace.buffer<double>(2, capacity * kMaxExercises * kMaxPayments, "CIR++ Bermudan bond a");
    auto* bond_b = workspace.buffer<double>(3, capacity * kMaxExercises * kMaxPayments, "CIR++ Bermudan bond b");
    const std::size_t state_capacity = capacity * kMaxExercises * num_paths;
    auto* immediate = workspace.buffer<double>(4, state_capacity, "CIR++ Bermudan immediate");
    auto* basis = workspace.buffer<double>(5, state_capacity, "CIR++ Bermudan basis");
    auto* discounts = workspace.buffer<double>(6, state_capacity, "CIR++ Bermudan discounts");
    auto* cashflows = workspace.buffer<double>(7, capacity * num_paths, "CIR++ Bermudan cashflows");
    auto* stats = workspace.buffer<double>(8, capacity * bermudan_lsm::kStatCount, "CIR++ Bermudan stats");
    auto* coefficients = workspace.buffer<double>(9, capacity * (bermudan_lsm::kBasisSize + 1U), "CIR++ Bermudan coefficients");
    auto* outputs = workspace.buffer<MonteCarloOutput>(10, capacity, "CIR++ Bermudan outputs");
    auto* step_coefficients = workspace.buffer<StepCoefficients>(
        11, capacity, "CIR++ Bermudan step coefficients"
    );
    float kernel_ms = 0.0F;
    for (std::size_t offset = 0; offset < row_count; offset += capacity) {
        const std::size_t remaining = row_count - offset;
        const std::size_t batch = remaining < capacity ? remaining : capacity;
        detail::check_cuda(cudaMemcpy(rows, host_rows + offset, batch * sizeof(*rows), cudaMemcpyHostToDevice), "copy CIR++ Bermudan rows");
        detail::check_cuda(cudaEventRecord(workspace.start_event()), "CIR++ Bermudan start");
        precompute_bonds_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(rows, batch, counts, bond_a, bond_b);
        precompute_steps_kernel<<<static_cast<unsigned int>(batch), 1>>>(
            rows, batch, target_dt, step_coefficients
        );
        simulate_exercise_values_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(
            rows, batch, num_paths, step_coefficients,
            bond_a, bond_b, immediate, basis, discounts
        );
        bermudan_lsm::initialize_cashflows_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(counts, immediate, discounts, batch, num_paths, cashflows);
        const auto shared = bermudan_lsm::kStatCount * detail::kThreadsPerBlock * sizeof(double);
        for (std::size_t exercise = kMaxExercises - 1U; exercise-- > 0U;) {
            bermudan_lsm::regression_stats_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock, shared>>>(
                counts, immediate, basis, discounts, cashflows, batch, num_paths, exercise, stats
            );
            bermudan_lsm::solve_kernel<<<static_cast<unsigned int>(batch), 1>>>(stats, batch, coefficients);
            bermudan_lsm::apply_exercise_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock>>>(
                counts, immediate, basis, discounts, coefficients, batch, num_paths, exercise, cashflows
            );
        }
        bermudan_lsm::finalize_kernel<<<static_cast<unsigned int>(batch), detail::kThreadsPerBlock, 2U * detail::kThreadsPerBlock * sizeof(double)>>>(cashflows, batch, num_paths, outputs);
        detail::check_cuda(cudaGetLastError(), "CIR++ Bermudan kernels");
        detail::check_cuda(cudaEventRecord(workspace.stop_event()), "CIR++ Bermudan stop");
        detail::check_cuda(cudaEventSynchronize(workspace.stop_event()), "CIR++ Bermudan sync");
        float elapsed = 0.0F;
        detail::check_cuda(cudaEventElapsedTime(&elapsed, workspace.start_event(), workspace.stop_event()), "CIR++ Bermudan timing");
        kernel_ms += elapsed;
        detail::check_cuda(cudaMemcpy(host_outputs + offset, outputs, batch * sizeof(*outputs), cudaMemcpyDeviceToHost), "copy CIR++ Bermudan outputs");
    }
    if (timing != nullptr) {
        timing->simulation_ms = kernel_ms;
        timing->total_ms = kernel_ms;
    }
}

}  // namespace ai_factory::cuda
