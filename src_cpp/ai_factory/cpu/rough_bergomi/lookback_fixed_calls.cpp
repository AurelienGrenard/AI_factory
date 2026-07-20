#include "ai_factory/cpu/rough_bergomi/lookback_fixed_calls.hpp"

#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/payoffs/lookback.hpp"
#include "ai_factory/cpu/rough_bergomi/common.hpp"

#include <algorithm>
#include <cmath>

namespace ai_factory::cpu::rough_bergomi {
namespace {

simulation::RoughBergomiModel to_model(const cuda::RoughBergomiRow& row) {
    return {
        row.spot,
        row.risk_free_rate,
        row.dividend_yield,
        row.forward_variance,
        row.eta,
        row.alpha,
        row.rho,
    };
}

simulation::TimeGrid to_time_grid(
    const cuda::RoughBergomiRow& row,
    std::size_t num_steps
) {
    return {row.maturity, num_steps};
}

simulation::SimulationConfig to_simulation(
    const cuda::RoughBergomiRow& row,
    std::size_t num_paths
) {
    return {
        row.seed,
        num_paths,
        simulation::kPhilox4x32_10BoxMuller,
    };
}

}  // namespace

void price_lookback_fixed_call(
    const cuda::RoughBergomiRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
) {
    const auto max_spots = simulation::generate_rough_bergomi_max_spots(
        to_model(row),
        to_time_grid(row, num_steps),
        to_simulation(row, num_paths)
    );

    const double discount = std::exp(-row.risk_free_rate * row.maturity);
    double sum = 0.0;
    double sumsq = 0.0;
    for (double max_spot : max_spots) {
        const double payoff =
            discount * payoffs::lookback_fixed_call(max_spot, row.strike);
        sum += payoff;
        sumsq += payoff * payoff;
    }
    const double path_count = static_cast<double>(num_paths);
    const double mean = sum / path_count;
    const double variance =
        (sumsq - path_count * mean * mean) / static_cast<double>(num_paths - 1U);
    output = {
        mean,
        std::sqrt(std::max(variance, 0.0)) / std::sqrt(path_count),
    };
}

void price_lookback_fixed_call_delta_crn(
    const cuda::RoughBergomiRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput& output
) {
    const auto max_spots = simulation::generate_rough_bergomi_max_spots(
        to_model(row),
        to_time_grid(row, num_steps),
        to_simulation(row, num_paths)
    );

    const double discount = std::exp(-row.risk_free_rate * row.maturity);
    double price_sum = 0.0;
    double price_sumsq = 0.0;
    double delta_sum = 0.0;
    double delta_sumsq = 0.0;
    for (double max_spot : max_spots) {
        const double price_payoff =
            discount * payoffs::lookback_fixed_call(max_spot, row.strike);
        const double up_payoff = discount
                                 * payoffs::lookback_fixed_call(
                                     max_spot * (1.0 + relative_bump),
                                     row.strike
                                 );
        const double down_payoff = discount
                                   * payoffs::lookback_fixed_call(
                                       max_spot * (1.0 - relative_bump),
                                       row.strike
                                   );
        const double delta =
            (up_payoff - down_payoff) / (2.0 * relative_bump * row.spot);
        price_sum += price_payoff;
        price_sumsq += price_payoff * price_payoff;
        delta_sum += delta;
        delta_sumsq += delta * delta;
    }

    const double path_count = static_cast<double>(num_paths);
    const double price = price_sum / path_count;
    const double price_variance =
        (price_sumsq - path_count * price * price)
        / static_cast<double>(num_paths - 1U);
    const double delta = delta_sum / path_count;
    const double delta_variance =
        (delta_sumsq - path_count * delta * delta)
        / static_cast<double>(num_paths - 1U);
    output = {
        price,
        std::sqrt(std::max(price_variance, 0.0)) / std::sqrt(path_count),
        delta,
        std::sqrt(std::max(delta_variance, 0.0)) / std::sqrt(path_count),
    };
}

void price_lookback_fixed_call_batch(
    const cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
) {
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(row_count);
         ++index) {
        price_lookback_fixed_call(
            rows[static_cast<std::size_t>(index)],
            num_paths,
            num_steps,
            outputs[static_cast<std::size_t>(index)]
        );
    }
}

void price_lookback_fixed_call_delta_crn_batch(
    const cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput* outputs
) {
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(row_count);
         ++index) {
        price_lookback_fixed_call_delta_crn(
            rows[static_cast<std::size_t>(index)],
            num_paths,
            num_steps,
            relative_bump,
            outputs[static_cast<std::size_t>(index)]
        );
    }
}

}  // namespace ai_factory::cpu::rough_bergomi
