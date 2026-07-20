#pragma once

#include "ai_factory/cpu/common/monte_carlo.hpp"
#include "ai_factory/cpu/common/time_grid.hpp"

#include <vector>

namespace ai_factory::simulation {

struct BlackScholesModel {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double volatility;
};

std::vector<double> generate_black_scholes_max_spots(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_black_scholes_arithmetic_average_spots(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_black_scholes_realized_volatilities(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_black_scholes_spot_paths(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_black_scholes_observation_spots(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count
);

}  // namespace ai_factory::simulation
