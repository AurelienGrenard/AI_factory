#include "ai_factory/cpu/black_scholes/lookback_fixed_calls.hpp"

#include "ai_factory/cpu/black_scholes/common.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/payoffs/lookback.hpp"

#include <algorithm>
#include <cmath>

namespace ai_factory::cpu::black_scholes {
namespace {

simulation::BlackScholesModel to_model(const cuda::BlackScholesRow& row) {
    return {row.spot, row.risk_free_rate, row.dividend_yield, row.volatility};
}

simulation::TimeGrid to_time_grid(const cuda::BlackScholesRow& row, std::size_t num_steps) {
    return {row.maturity, num_steps};
}

simulation::SimulationConfig to_simulation(const cuda::BlackScholesRow& row, std::size_t num_paths) {
    return {row.seed, num_paths, simulation::kPhilox4x32_10BoxMuller};
}

void summarize(double sum, double sumsq, std::size_t num_paths, cuda::MonteCarloOutput& output) {
    const double path_count = static_cast<double>(num_paths);
    const double mean = sum / path_count;
    const double variance = (sumsq - path_count * mean * mean)
                            / static_cast<double>(num_paths - 1U);
    output = {mean, std::sqrt(std::max(variance, 0.0)) / std::sqrt(path_count)};
}

}  // namespace

void price_lookback_fixed_call_batch(
    const cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(row_count); ++index) {
        const auto& row = rows[static_cast<std::size_t>(index)];
        const auto max_spots = simulation::generate_black_scholes_max_spots(
            to_model(row), to_time_grid(row, num_steps), to_simulation(row, num_paths)
        );
        const double discount = std::exp(-row.risk_free_rate * row.maturity);
        double sum = 0.0;
        double sumsq = 0.0;
        for (double max_spot : max_spots) {
            const double payoff = discount * payoffs::lookback_fixed_call(max_spot, row.strike);
            sum += payoff;
            sumsq += payoff * payoff;
        }
        summarize(sum, sumsq, num_paths, outputs[static_cast<std::size_t>(index)]);
    }
}

void price_lookback_fixed_call_delta_crn_batch(
    const cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(row_count); ++index) {
        const auto& row = rows[static_cast<std::size_t>(index)];
        const auto max_spots = simulation::generate_black_scholes_max_spots(
            to_model(row), to_time_grid(row, num_steps), to_simulation(row, num_paths)
        );
        const double discount = std::exp(-row.risk_free_rate * row.maturity);
        double price_sum = 0.0;
        double price_sumsq = 0.0;
        double delta_sum = 0.0;
        double delta_sumsq = 0.0;
        for (double max_spot : max_spots) {
            const double price = discount * payoffs::lookback_fixed_call(max_spot, row.strike);
            const double up = discount
                              * payoffs::lookback_fixed_call(max_spot * (1.0 + relative_bump), row.strike);
            const double down = discount
                                * payoffs::lookback_fixed_call(max_spot * (1.0 - relative_bump), row.strike);
            const double delta = (up - down) / (2.0 * relative_bump * row.spot);
            price_sum += price;
            price_sumsq += price * price;
            delta_sum += delta;
            delta_sumsq += delta * delta;
        }
        const double path_count = static_cast<double>(num_paths);
        const double price = price_sum / path_count;
        const double price_variance =
            (price_sumsq - path_count * price * price) / static_cast<double>(num_paths - 1U);
        const double delta = delta_sum / path_count;
        const double delta_variance =
            (delta_sumsq - path_count * delta * delta) / static_cast<double>(num_paths - 1U);
        outputs[static_cast<std::size_t>(index)] = {
            price,
            std::sqrt(std::max(price_variance, 0.0)) / std::sqrt(path_count),
            delta,
            std::sqrt(std::max(delta_variance, 0.0)) / std::sqrt(path_count),
        };
    }
}

}  // namespace ai_factory::cpu::black_scholes
