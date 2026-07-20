#include "ai_factory/cpu/cir/swaptions.hpp"

#include "ai_factory/common/fixed_income/cir.hpp"
#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ai_factory::cpu::cir {
namespace {

constexpr int kMaxPayments = 20;
constexpr double kQePsiCutoff = 1.5;

}  // namespace

void price_swaption_batch(
    const cuda::CirSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    double target_dt,
    cuda::MonteCarloOutput* outputs
) {
    if (num_paths < 2U) {
        throw std::invalid_argument("Swaption pricing requires at least two paths.");
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
        const std::size_t num_steps = std::max<std::size_t>(
            1U,
            static_cast<std::size_t>(std::llround(product.expiry / target_dt))
        );
        const double dt = product.expiry / static_cast<double>(num_steps);
        const double decay = std::exp(-model.kappa * dt);
        const double one_minus_decay = 1.0 - decay;
        const double volatility_squared =
            model.volatility * model.volatility;

        std::array<double, kMaxPayments> bond_a{};
        std::array<double, kMaxPayments> bond_b{};
        for (int payment = 0; payment < product.payment_count; ++payment) {
            const double horizon = (payment + 1) * product.accrual_period;
            bond_a[payment] = fixed_income::cir_bond_a(
                model.kappa, model.theta, model.volatility, horizon
            );
            bond_b[payment] = fixed_income::cir_bond_b(
                model.kappa, model.volatility, horizon
            );
        }

        double sum = 0.0;
        double sumsq = 0.0;
        for (std::size_t path = 0; path < num_paths; ++path) {
            double rate = model.initial_rate;
            double integral = 0.0;
            simulation::PhiloxNormalSequence normals(row.seed, path);
            simulation::PhiloxUniformSequence uniforms(
                row.seed, path + num_paths
            );
            for (std::size_t step = 0; step < num_steps; ++step) {
                const double previous_rate = rate;
                const double mean =
                    model.theta + (rate - model.theta) * decay;
                const double variance =
                    rate * volatility_squared * decay * one_minus_decay
                        / model.kappa
                    + model.theta * volatility_squared
                          * one_minus_decay * one_minus_decay
                          / (2.0 * model.kappa);
                const double psi = variance / (mean * mean);
                const double normal = normals.next();
                const double uniform = uniforms.next();

                if (psi <= kQePsiCutoff) {
                    const double inverse_psi = 1.0 / psi;
                    const double b_squared =
                        2.0 * inverse_psi - 1.0
                        + std::sqrt(2.0 * inverse_psi)
                              * std::sqrt(std::max(
                                  2.0 * inverse_psi - 1.0, 0.0
                              ));
                    const double scale = mean / (1.0 + b_squared);
                    const double shifted = std::sqrt(b_squared) + normal;
                    rate = scale * shifted * shifted;
                } else {
                    const double probability = (psi - 1.0) / (psi + 1.0);
                    const double beta = (1.0 - probability) / mean;
                    rate = uniform <= probability
                               ? 0.0
                               : std::log(
                                     (1.0 - probability) / (1.0 - uniform)
                                 )
                                     / beta;
                }
                integral += 0.5 * (previous_rate + rate) * dt;
            }

            double annuity = 0.0;
            for (int payment = 0;
                 payment < product.payment_count;
                 ++payment) {
                annuity += product.accrual_period * bond_a[payment]
                           * std::exp(-bond_b[payment] * rate);
            }
            const int last_payment = product.payment_count - 1;
            const double end_bond = bond_a[last_payment]
                                    * std::exp(-bond_b[last_payment] * rate);
            const double swap = static_cast<double>(product.direction)
                                * (1.0 - end_bond
                                   - product.fixed_rate * annuity);
            const double payoff = std::exp(-integral) * product.notional
                                  * std::max(swap, 0.0);
            sum += payoff;
            sumsq += payoff * payoff;
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

}  // namespace ai_factory::cpu::cir
