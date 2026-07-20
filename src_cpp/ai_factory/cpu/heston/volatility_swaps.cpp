#include "ai_factory/cpu/heston/volatility_swaps.hpp"

#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/heston/common.hpp"
#include "ai_factory/cpu/common/payoffs/volatility_swap.hpp"

#include <algorithm>
#include <cmath>

namespace ai_factory::cpu::heston {
namespace {

constexpr double kObservationsPerYear = 52.0;

simulation::HestonModel to_model(const cuda::HestonRow& row) {
    return {
        row.spot,
        row.risk_free_rate,
        row.dividend_yield,
        row.initial_variance,
        row.kappa,
        row.theta,
        row.volatility_of_variance,
        row.rho,
    };
}

simulation::TimeGrid to_time_grid(
    const cuda::HestonRow& row,
    std::size_t num_steps
) {
    return {row.maturity, num_steps};
}

simulation::SimulationConfig to_simulation(
    const cuda::HestonRow& row,
    std::size_t num_paths
) {
    return {
        row.seed,
        num_paths,
        simulation::kPhilox4x32_10BoxMuller,
    };
}

simulation::HestonSimulationScheme to_scheme(const cuda::HestonRow& row) {
    return static_cast<simulation::HestonSimulationScheme>(row.scheme);
}

double standard_error(double sumsq, double mean, std::size_t num_paths) {
    const double path_count = static_cast<double>(num_paths);
    const double variance =
        (sumsq - path_count * mean * mean) / static_cast<double>(num_paths - 1U);
    return std::sqrt(std::max(variance, 0.0)) / std::sqrt(path_count);
}

}  // namespace

void price_volatility_swap(
    const cuda::HestonRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
) {
    const auto realized_volatilities =
        simulation::generate_heston_realized_volatilities(
            to_model(row),
            to_time_grid(row, num_steps),
            to_simulation(row, num_paths),
            to_scheme(row),
            kObservationsPerYear
        );

    const double discount = std::exp(-row.risk_free_rate * row.maturity);
    double sum = 0.0;
    double sumsq = 0.0;
    for (double realized_volatility : realized_volatilities) {
        const double payoff =
            discount * payoffs::volatility_swap(realized_volatility, row.strike);
        sum += payoff;
        sumsq += payoff * payoff;
    }
    const double mean = sum / static_cast<double>(num_paths);
    output = {mean, standard_error(sumsq, mean, num_paths)};
}

void price_volatility_swap_batch(
    const cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(row_count);
         ++index) {
        price_volatility_swap(
            rows[static_cast<std::size_t>(index)],
            num_paths,
            num_steps,
            outputs[static_cast<std::size_t>(index)]
        );
    }
}

}  // namespace ai_factory::cpu::heston
