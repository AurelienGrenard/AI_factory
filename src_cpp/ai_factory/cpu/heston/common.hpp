#pragma once

#include "ai_factory/cpu/common/monte_carlo.hpp"
#include "ai_factory/cpu/common/time_grid.hpp"

#include <string>
#include <vector>

namespace ai_factory::simulation {

struct HestonModel {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double initial_variance;
    double kappa;
    double theta;
    double volatility_of_variance;
    double rho;
};

struct HestonStatePaths {
    std::vector<double> spots;
    std::vector<double> variances;
};

enum class HestonSimulationScheme {
    EulerFullTruncation = 0,
    AndersenQe = 1,
    AndersenQeMartingale = 2,
};

HestonSimulationScheme parse_heston_simulation_scheme(const std::string& value);

std::vector<double> generate_heston_terminal_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation
);

TimedTerminalSpots generate_heston_terminal_spots_timed(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation
);

std::vector<double> generate_heston_max_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation
);

std::vector<double> generate_heston_arithmetic_average_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation
);

std::vector<double> generate_heston_realized_volatilities(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation,
    double observations_per_year = 52.0
);

std::vector<double> generate_heston_spot_paths(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation
);

HestonStatePaths generate_heston_state_paths(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation
);

std::vector<double> generate_heston_observation_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count,
    HestonSimulationScheme scheme = HestonSimulationScheme::EulerFullTruncation
);

}  // namespace ai_factory::simulation
