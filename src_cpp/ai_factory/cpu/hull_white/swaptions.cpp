#include "ai_factory/cpu/hull_white/swaptions.hpp"

#include "ai_factory/common/fixed_income/hull_white.hpp"
#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ai_factory::cpu::hull_white {
namespace {

constexpr int kMaxPayments = 20;

}  // namespace

void price_swaption_batch(
    const cuda::HullWhiteSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    cuda::MonteCarloOutput* outputs
) {
    if (num_paths < 2U) throw std::invalid_argument("Swaption pricing requires at least two paths.");
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(row_count); ++index) {
        const auto& row = rows[index];
        const auto& product = row.product;
        const auto normals = simulation::philox_standard_normals(
            row.seed, 2U * num_paths, 0U
        );
        const double state_var = fixed_income::hull_white_state_variance(row.mean_reversion, row.volatility, product.expiry);
        const double integral_var = fixed_income::hull_white_integral_variance(row.mean_reversion, row.volatility, product.expiry);
        const double covariance = fixed_income::hull_white_state_integral_covariance(row.mean_reversion, row.volatility, product.expiry);
        const double state_scale = std::sqrt(state_var);
        const double integral_loading = covariance / state_scale;
        const double residual_scale = std::sqrt(std::max(integral_var - integral_loading * integral_loading, 0.0));
        const double deterministic_integral = fixed_income::hull_white_deterministic_integral(
            product.expiry, row.mean_reversion, row.volatility,
            row.beta0, row.beta1, row.beta2, row.tau
        );
        std::array<double, kMaxPayments> bond_a{};
        std::array<double, kMaxPayments> bond_b{};
        for (int payment = 0; payment < product.payment_count; ++payment) {
            const double maturity =
                product.expiry + (payment + 1) * product.accrual_period;
            bond_a[payment] = fixed_income::hull_white_bond_a(
                product.expiry,
                maturity,
                row.mean_reversion,
                row.volatility,
                row.beta0,
                row.beta1,
                row.beta2,
                row.tau
            );
            bond_b[payment] = fixed_income::hull_white_b(
                row.mean_reversion, maturity - product.expiry
            );
        }
        double sum = 0.0, sumsq = 0.0;
        for (std::size_t path = 0; path < num_paths; ++path) {
            const double first_normal = normals[2U * path];
            const double second_normal = normals[2U * path + 1U];
            const double state = state_scale * first_normal;
            const double state_integral = integral_loading * first_normal
                                          + residual_scale * second_normal;
            double annuity = 0.0;
            for (int payment = 0; payment < product.payment_count; ++payment) {
                annuity += product.accrual_period * bond_a[payment]
                           * std::exp(-bond_b[payment] * state);
            }
            const int last_payment = product.payment_count - 1;
            const double end_bond = bond_a[last_payment]
                                    * std::exp(-bond_b[last_payment] * state);
            const double swap = static_cast<double>(product.direction)
                                * (1.0 - end_bond - product.fixed_rate * annuity);
            const double payoff = std::exp(-deterministic_integral - state_integral)
                                  * product.notional * std::max(swap, 0.0);
            sum += payoff; sumsq += payoff * payoff;
        }
        const double n = static_cast<double>(num_paths);
        const double mean = sum / n;
        const double variance = (sumsq - n * mean * mean) / (n - 1.0);
        outputs[index] = {mean, std::sqrt(std::max(variance, 0.0) / n)};
    }
}

}
