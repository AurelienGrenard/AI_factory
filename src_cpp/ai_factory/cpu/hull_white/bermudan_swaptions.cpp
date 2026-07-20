#include "ai_factory/cpu/hull_white/bermudan_swaptions.hpp"

#include "ai_factory/common/fixed_income/hull_white.hpp"
#include "ai_factory/cpu/common/bermudan_lsm.hpp"
#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ai_factory::cpu::hull_white {
namespace {

constexpr int kMaxExercises = 8;
constexpr int kMaxPayments = 20;
constexpr double kBasisRateScale = 0.04;

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
    const cuda::HullWhiteBermudanSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    cuda::MonteCarloOutput* outputs
) {
    if (num_paths < 2U) {
        throw std::invalid_argument("Bermudan swaption pricing requires two paths.");
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t row_index = 0;
         row_index < static_cast<std::ptrdiff_t>(row_count);
         ++row_index) {
        const auto& row = rows[row_index];
        const auto& product = row.product;
        validate(product);
        const auto exercise_count = static_cast<std::size_t>(
            product.exercise_count
        );

        std::array<std::array<double, kMaxPayments>, kMaxExercises> bond_a{};
        std::array<std::array<double, kMaxPayments>, kMaxExercises> bond_b{};
        std::array<double, kMaxExercises> exercise_times{};
        std::array<double, kMaxExercises> decay{};
        std::array<double, kMaxExercises> state_scale{};
        std::array<double, kMaxExercises> integral_loading{};
        std::array<double, kMaxExercises> residual_scale{};
        std::array<double, kMaxExercises> integral_state_loading{};
        std::array<double, kMaxExercises> deterministic_integral{};

        double previous_time = 0.0;
        for (std::size_t exercise = 0; exercise < exercise_count; ++exercise) {
            const double time = product.first_exercise
                                + exercise * product.exercise_period;
            exercise_times[exercise] = time;
            const double interval = time - previous_time;
            decay[exercise] = std::exp(-row.mean_reversion * interval);
            state_scale[exercise] = std::sqrt(
                fixed_income::hull_white_state_variance(
                    row.mean_reversion, row.volatility, interval
                )
            );
            const double integral_variance =
                fixed_income::hull_white_integral_variance(
                    row.mean_reversion, row.volatility, interval
                );
            const double covariance =
                fixed_income::hull_white_state_integral_covariance(
                    row.mean_reversion, row.volatility, interval
                );
            integral_loading[exercise] = covariance / state_scale[exercise];
            residual_scale[exercise] = std::sqrt(std::max(
                integral_variance
                    - integral_loading[exercise] * integral_loading[exercise],
                0.0
            ));
            integral_state_loading[exercise] = fixed_income::hull_white_b(
                row.mean_reversion, interval
            );
            deterministic_integral[exercise] =
                fixed_income::hull_white_deterministic_integral(
                    time,
                    row.mean_reversion,
                    row.volatility,
                    row.beta0,
                    row.beta1,
                    row.beta2,
                    row.tau
                );
            for (int payment = static_cast<int>(exercise);
                 payment < product.payment_count;
                 ++payment) {
                const double maturity = product.first_exercise
                                        + (payment + 1)
                                              * product.accrual_period;
                bond_a[exercise][payment] = fixed_income::hull_white_bond_a(
                    time,
                    maturity,
                    row.mean_reversion,
                    row.volatility,
                    row.beta0,
                    row.beta1,
                    row.beta2,
                    row.tau
                );
                bond_b[exercise][payment] = fixed_income::hull_white_b(
                    row.mean_reversion, maturity - time
                );
            }
            previous_time = time;
        }

        std::vector<double> states(exercise_count * num_paths);
        std::vector<double> discounts(exercise_count * num_paths);
        for (std::size_t path = 0; path < num_paths; ++path) {
            simulation::PhiloxNormalSequence normals(
                row.seed, 0U, path * 2U * exercise_count
            );
            double state = 0.0;
            double stochastic_integral = 0.0;
            for (std::size_t exercise = 0;
                 exercise < exercise_count;
                 ++exercise) {
                const double previous_state = state;
                const double first_normal = normals.next();
                const double second_normal = normals.next();
                state = decay[exercise] * previous_state
                        + state_scale[exercise] * first_normal;
                stochastic_integral +=
                    integral_state_loading[exercise] * previous_state
                    + integral_loading[exercise] * first_normal
                    + residual_scale[exercise] * second_normal;
                const std::size_t index = exercise * num_paths + path;
                states[index] = state;
                discounts[index] = std::exp(
                    -deterministic_integral[exercise] - stochastic_integral
                );
            }
        }

        const auto evaluate = [&](std::size_t exercise, double state) {
            double annuity = 0.0;
            for (int payment = static_cast<int>(exercise);
                 payment < product.payment_count;
                 ++payment) {
                annuity += product.accrual_period * bond_a[exercise][payment]
                           * std::exp(-bond_b[exercise][payment] * state);
            }
            const int last_payment = product.payment_count - 1;
            const double end_bond = bond_a[exercise][last_payment]
                                    * std::exp(
                                        -bond_b[exercise][last_payment] * state
                                    );
            const double signed_swap = static_cast<double>(product.direction)
                                       * (1.0 - end_bond
                                          - product.fixed_rate * annuity);
            const double par_rate = (1.0 - end_bond) / annuity;
            return lsm::ExerciseValue{
                product.notional * std::max(signed_swap, 0.0),
                par_rate / kBasisRateScale,
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
        const double path_count = static_cast<double>(num_paths);
        const double mean = sum / path_count;
        const double variance =
            (sumsq - path_count * mean * mean) / (path_count - 1.0);
        outputs[row_index] = {
            mean,
            std::sqrt(std::max(variance, 0.0) / path_count),
        };
    }
}

}  // namespace ai_factory::cpu::hull_white
