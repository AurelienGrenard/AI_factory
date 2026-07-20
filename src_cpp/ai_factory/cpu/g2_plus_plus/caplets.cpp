#include "ai_factory/cpu/g2_plus_plus/caplets.hpp"

#include "ai_factory/common/fixed_income/g2_plus_plus.hpp"
#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ai_factory::cpu::g2_plus_plus {

void price_caplet_batch(
    const cuda::G2PlusPlusCapletRow* rows,
    std::size_t count,
    std::size_t paths,
    cuda::MonteCarloOutput* outputs
) {
    if (paths < 2U) {
        throw std::invalid_argument("Caplet pricing requires at least two paths.");
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t signed_index = 0;
         signed_index < static_cast<std::ptrdiff_t>(count);
         ++signed_index) {
        const auto& row = rows[signed_index];
        const auto& model = row.model;
        const auto& product = row.product;
        const auto transition = fixed_income::make_g2_transition(
            model.mean_reversion_x, model.volatility_x,
            model.mean_reversion_y, model.volatility_y, model.rho,
            product.fixing_time
        );
        const double maturity = product.fixing_time + product.accrual_period;
        const double bond_a = fixed_income::g2_bond_price(
            0.0, 0.0, product.fixing_time, maturity,
            model.mean_reversion_x, model.volatility_x,
            model.mean_reversion_y, model.volatility_y, model.rho,
            row.curve.beta0, row.curve.beta1, row.curve.beta2, row.curve.tau
        );
        const double bond_x = fixed_income::ou_b(
            model.mean_reversion_x, product.accrual_period
        );
        const double bond_y = fixed_income::ou_b(
            model.mean_reversion_y, product.accrual_period
        );
        const double discount_a = fixed_income::nelson_siegel_discount(
            product.fixing_time,
            row.curve.beta0, row.curve.beta1, row.curve.beta2, row.curve.tau
        ) * std::exp(-0.5 * fixed_income::g2_integral_variance(
            model.mean_reversion_x, model.volatility_x,
            model.mean_reversion_y, model.volatility_y, model.rho,
            product.fixing_time
        ));
        simulation::PhiloxNormalSequence normals(row.seed, 0U, 0U);
        double sum = 0.0;
        double sumsq = 0.0;
        for (std::size_t path = 0; path < paths; ++path) {
            const double z0 = normals.next();
            const double z1 = normals.next();
            const double z2 = normals.next();
            const double z3 = normals.next();
            double x = 0.0;
            double y = 0.0;
            double integrated_x = 0.0;
            double integrated_y = 0.0;
            fixed_income::apply_g2_transition(
                transition,
                z0, z1, z2, z3,
                x, y, integrated_x, integrated_y
            );
            const double bond = bond_a * std::exp(-bond_x * x - bond_y * y);
            const double discount = discount_a
                                    * std::exp(-integrated_x - integrated_y);
            const double payoff = discount * product.notional * std::max(
                1.0 - (1.0 + product.accrual_period * product.strike) * bond,
                0.0
            );
            sum += payoff;
            sumsq += payoff * payoff;
        }
        const double count_as_double = static_cast<double>(paths);
        const double mean = sum / count_as_double;
        const double variance = (
            sumsq - count_as_double * mean * mean
        ) / (count_as_double - 1.0);
        outputs[signed_index] = {
            mean,
            std::sqrt(std::max(variance, 0.0) / count_as_double),
        };
    }
}

}  // namespace ai_factory::cpu::g2_plus_plus
