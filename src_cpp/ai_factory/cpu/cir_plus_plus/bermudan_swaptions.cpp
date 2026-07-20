#include "ai_factory/cpu/cir_plus_plus/bermudan_swaptions.hpp"

#include "ai_factory/common/fixed_income/cir_plus_plus.hpp"
#include "ai_factory/cpu/common/bermudan_lsm.hpp"
#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ai_factory::cpu::cir_plus_plus {
namespace {

constexpr int kMaxExercises = 8;
constexpr int kMaxPayments = 20;
constexpr double kBasisRateScale = 0.04;
constexpr double kQePsiCutoff = 1.5;

void validate(const cuda::BermudanSwaptionTerms& product) {
    if (product.exercise_count < 2 || product.exercise_count > kMaxExercises
        || product.payment_count < product.exercise_count + 2
        || product.payment_count > kMaxPayments
        || std::abs(product.exercise_period - product.accrual_period) > 1.0e-12) {
        throw std::invalid_argument("Invalid Bermudan swaption schedule.");
    }
}

}  // namespace

void price_bermudan_swaption_batch(
    const cuda::CirPlusPlusBermudanSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    double target_dt,
    cuda::MonteCarloOutput* outputs
) {
    if (num_paths < 2U || target_dt <= 0.0) {
        throw std::invalid_argument("Invalid CIR Bermudan Monte Carlo settings.");
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t row_index = 0;
         row_index < static_cast<std::ptrdiff_t>(row_count);
         ++row_index) {
        const auto& row = rows[row_index];
        const auto& model = row.model;
        const auto& product = row.product;
        validate(product);
        const auto exercise_count = static_cast<std::size_t>(product.exercise_count);
        const double last_exercise = product.first_exercise
            + (product.exercise_count - 1) * product.exercise_period;
        const auto num_steps = static_cast<std::size_t>(std::max(
            1LL, std::llround(last_exercise / target_dt)
        ));
        const double dt = last_exercise / static_cast<double>(num_steps);

        std::array<std::size_t, kMaxExercises> exercise_steps{};
        std::array<std::array<double, kMaxPayments>, kMaxExercises> bond_a{};
        std::array<std::array<double, kMaxPayments>, kMaxExercises> bond_b{};
        for (std::size_t exercise = 0; exercise < exercise_count; ++exercise) {
            const double exercise_time = product.first_exercise
                + exercise * product.exercise_period;
            exercise_steps[exercise] = static_cast<std::size_t>(
                std::llround(exercise_time / dt)
            );
            for (int payment = static_cast<int>(exercise);
                 payment < product.payment_count;
                 ++payment) {
                const double maturity = product.first_exercise
                    + (payment + 1) * product.accrual_period;
                bond_a[exercise][payment] = fixed_income::cir_plus_plus_bond_a(
                    exercise_time, maturity, model.initial_factor,
                    model.kappa, model.theta, model.volatility,
                    row.curve.beta0, row.curve.beta1,
                    row.curve.beta2, row.curve.tau
                );
                bond_b[exercise][payment] = fixed_income::cir_bond_b(
                    model.kappa, model.volatility, maturity - exercise_time
                );
            }
        }

        const double decay = std::exp(-model.kappa * dt);
        const double one_minus_decay = 1.0 - decay;
        const double volatility_squared = model.volatility * model.volatility;
        std::vector<double> states(exercise_count * num_paths);
        std::vector<double> discounts(exercise_count * num_paths);
        for (std::size_t path = 0; path < num_paths; ++path) {
            simulation::PhiloxNormalSequence normals(row.seed, path);
            simulation::PhiloxUniformSequence uniforms(row.seed, path + num_paths);
            double rate = model.initial_factor;
            double integral = 0.0;
            std::size_t next_exercise = 0U;
            for (std::size_t step = 1U; step <= num_steps; ++step) {
                const double previous_rate = rate;
                const double mean = model.theta + (rate - model.theta) * decay;
                const double variance =
                    rate * volatility_squared * decay * one_minus_decay / model.kappa
                    + model.theta * volatility_squared * one_minus_decay
                          * one_minus_decay / (2.0 * model.kappa);
                const double psi = variance / (mean * mean);
                const double normal = normals.next();
                const double uniform = uniforms.next();
                if (psi <= kQePsiCutoff) {
                    const double inverse_psi = 1.0 / psi;
                    const double b_squared = 2.0 * inverse_psi - 1.0
                        + std::sqrt(2.0 * inverse_psi)
                              * std::sqrt(std::max(2.0 * inverse_psi - 1.0, 0.0));
                    const double shifted = std::sqrt(b_squared) + normal;
                    rate = mean / (1.0 + b_squared) * shifted * shifted;
                } else {
                    const double probability = (psi - 1.0) / (psi + 1.0);
                    rate = uniform <= probability
                        ? 0.0
                        : -mean * std::log((1.0 - uniform) / (1.0 - probability))
                              / (1.0 - probability);
                }
                integral += 0.5 * (previous_rate + rate) * dt;
                if (next_exercise < exercise_count
                    && step == exercise_steps[next_exercise]) {
                    const std::size_t index = next_exercise * num_paths + path;
                    states[index] = rate;
                    const double exercise_time = product.first_exercise
                        + next_exercise * product.exercise_period;
                    discounts[index] = fixed_income::cir_plus_plus_path_discount(
                        integral, exercise_time, model.initial_factor,
                        model.kappa, model.theta, model.volatility,
                        row.curve.beta0, row.curve.beta1,
                        row.curve.beta2, row.curve.tau
                    );
                    ++next_exercise;
                }
            }
        }

        const auto evaluate = [&](std::size_t exercise, double rate) {
            double annuity = 0.0;
            for (int payment = static_cast<int>(exercise);
                 payment < product.payment_count;
                 ++payment) {
                annuity += product.accrual_period * bond_a[exercise][payment]
                    * std::exp(-bond_b[exercise][payment] * rate);
            }
            const int last_payment = product.payment_count - 1;
            const double end_bond = bond_a[exercise][last_payment]
                * std::exp(-bond_b[exercise][last_payment] * rate);
            const double signed_swap = static_cast<double>(product.direction)
                * (1.0 - end_bond - product.fixed_rate * annuity);
            return lsm::ExerciseValue{
                product.notional * std::max(signed_swap, 0.0),
                ((1.0 - end_bond) / annuity) / kBasisRateScale,
            };
        };
        const auto cashflows = lsm::bermudan_cashflows_present_value(
            states, discounts, exercise_count, num_paths, evaluate
        );
        double sum = 0.0;
        double sumsq = 0.0;
        for (const double cashflow : cashflows) {
            sum += cashflow;
            sumsq += cashflow * cashflow;
        }
        const double n = static_cast<double>(num_paths);
        const double mean = sum / n;
        const double variance = (sumsq - n * mean * mean) / (n - 1.0);
        outputs[row_index] = {mean, std::sqrt(std::max(variance, 0.0) / n)};
    }
}

}  // namespace ai_factory::cpu::cir_plus_plus
