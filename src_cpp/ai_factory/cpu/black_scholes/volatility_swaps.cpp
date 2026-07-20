#include "ai_factory/cpu/black_scholes/volatility_swaps.hpp"

#include "ai_factory/cpu/black_scholes/common.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/payoffs/volatility_swap.hpp"

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

}  // namespace

void price_volatility_swap_batch(
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
        const auto realized_vols = simulation::generate_black_scholes_realized_volatilities(
            to_model(row), to_time_grid(row, num_steps), to_simulation(row, num_paths)
        );
        const double discount = std::exp(-row.risk_free_rate * row.maturity);
        double sum = 0.0;
        double sumsq = 0.0;
        for (double realized_vol : realized_vols) {
            const double payoff = discount * payoffs::volatility_swap(realized_vol, row.strike);
            sum += payoff;
            sumsq += payoff * payoff;
        }
        const double path_count = static_cast<double>(num_paths);
        const double mean = sum / path_count;
        const double variance = (sumsq - path_count * mean * mean)
                                / static_cast<double>(num_paths - 1U);
        outputs[static_cast<std::size_t>(index)] = {
            mean,
            std::sqrt(std::max(variance, 0.0)) / std::sqrt(path_count),
        };
    }
}

}  // namespace ai_factory::cpu::black_scholes
