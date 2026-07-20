#pragma once

#include "ai_factory/cpu/common/monte_carlo.hpp"
#include "ai_factory/cpu/common/time_grid.hpp"

#include <cstddef>
#include <vector>

namespace ai_factory::simulation {

struct RoughHestonModel {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double initial_variance;
    double kappa;
    double theta;
    double volatility_of_variance;
    double hurst;
    double rho;
};

constexpr std::size_t kRoughHestonFactorCount = 8U;

struct RoughHestonStatePaths {
    std::vector<double> spots;
    std::vector<double> factors;
};

using RoughHestonObservationVisitor = bool (*)(
    std::size_t, std::size_t, double, void*
);

std::vector<double> generate_rough_heston_terminal_spots(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&
);
std::vector<double> generate_rough_heston_max_spots(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&
);
std::vector<double> generate_rough_heston_arithmetic_average_spots(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&
);
std::vector<double> generate_rough_heston_realized_volatilities(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&,
    double observations_per_year = 52.0
);
std::vector<double> generate_rough_heston_spot_paths(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&
);
RoughHestonStatePaths generate_rough_heston_state_paths(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&
);
std::vector<double> generate_rough_heston_observation_spots(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&,
    std::size_t observation_count
);
void visit_rough_heston_observation_spots(
    const RoughHestonModel&, const TimeGrid&, const SimulationConfig&,
    std::size_t observation_count, RoughHestonObservationVisitor, void*
);

}  // namespace ai_factory::simulation
