#include "ai_factory/cpu/rough_heston/asian_arithmetic_calls.hpp"

#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/payoffs/asian.hpp"
#include "ai_factory/cpu/rough_heston/common.hpp"

#include <algorithm>
#include <cmath>

namespace ai_factory::cpu::rough_heston {
namespace {

simulation::RoughHestonModel to_model(const cuda::RoughHestonRow& row) {
    return {
        row.spot,
        row.risk_free_rate,
        row.dividend_yield,
        row.initial_variance,
        row.kappa,
        row.theta,
        row.volatility_of_variance,
        row.hurst,
        row.rho,
    };
}

simulation::TimeGrid to_time_grid(
    const cuda::RoughHestonRow& row,
    std::size_t num_steps
) {
    return {row.maturity, num_steps};
}

simulation::SimulationConfig to_simulation(
    const cuda::RoughHestonRow& row,
    std::size_t num_paths
) {
    return {
        row.seed,
        num_paths,
        simulation::kPhilox4x32_10BoxMuller,
    };
}

double standard_error(double sumsq, double mean, std::size_t num_paths) {
    const double path_count = static_cast<double>(num_paths);
    const double variance =
        (sumsq - path_count * mean * mean) / static_cast<double>(num_paths - 1U);
    return std::sqrt(std::max(variance, 0.0)) / std::sqrt(path_count);
}

}  // namespace

void price_asian_arithmetic_call(
    const cuda::RoughHestonRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput& output
) {
    const auto average_spots =
        simulation::generate_rough_heston_arithmetic_average_spots(
            to_model(row),
            to_time_grid(row, num_steps),
            to_simulation(row, num_paths)
        );

    const double discount = std::exp(-row.risk_free_rate * row.maturity);
    double sum = 0.0;
    double sumsq = 0.0;
    for (double average_spot : average_spots) {
        const double payoff =
            discount * payoffs::asian_arithmetic_call(average_spot, row.strike);
        sum += payoff;
        sumsq += payoff * payoff;
    }
    const double mean = sum / static_cast<double>(num_paths);
    output = {mean, standard_error(sumsq, mean, num_paths)};
}

void price_asian_arithmetic_call_delta_crn(
    const cuda::RoughHestonRow& row,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput& output
) {
    const auto average_spots =
        simulation::generate_rough_heston_arithmetic_average_spots(
            to_model(row),
            to_time_grid(row, num_steps),
            to_simulation(row, num_paths)
        );

    const double discount = std::exp(-row.risk_free_rate * row.maturity);
    double price_sum = 0.0;
    double price_sumsq = 0.0;
    double delta_sum = 0.0;
    double delta_sumsq = 0.0;
    for (double average_spot : average_spots) {
        const double price_payoff =
            discount * payoffs::asian_arithmetic_call(average_spot, row.strike);
        const double up_payoff =
            discount
            * payoffs::asian_arithmetic_call(
                average_spot * (1.0 + relative_bump),
                row.strike
            );
        const double down_payoff =
            discount
            * payoffs::asian_arithmetic_call(
                average_spot * (1.0 - relative_bump),
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
    const double delta = delta_sum / path_count;
    output = {
        price,
        standard_error(price_sumsq, price, num_paths),
        delta,
        standard_error(delta_sumsq, delta, num_paths),
    };
}

void price_asian_arithmetic_call_batch(
    const cuda::RoughHestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
) {
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(row_count);
         ++index) {
        price_asian_arithmetic_call(
            rows[static_cast<std::size_t>(index)],
            num_paths,
            num_steps,
            outputs[static_cast<std::size_t>(index)]
        );
    }
}

void price_asian_arithmetic_call_delta_crn_batch(
    const cuda::RoughHestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    cuda::PriceDeltaOutput* outputs
) {
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(row_count);
         ++index) {
        price_asian_arithmetic_call_delta_crn(
            rows[static_cast<std::size_t>(index)],
            num_paths,
            num_steps,
            relative_bump,
            outputs[static_cast<std::size_t>(index)]
        );
    }
}

}  // namespace ai_factory::cpu::rough_heston
