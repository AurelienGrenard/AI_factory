#pragma once

#include "ai_factory/cpu/common/monte_carlo.hpp"
#include "ai_factory/cpu/common/time_grid.hpp"

#include <vector>

namespace ai_factory::simulation {

struct RoughBergomiModel {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double forward_variance;
    double eta;
    double alpha;
    double rho;
};

using RoughBergomiObservationVisitor = bool (*)(
    std::size_t path,
    std::size_t observation,
    double spot,
    void* context
);

std::vector<double> generate_rough_bergomi_max_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_rough_bergomi_terminal_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_rough_bergomi_arithmetic_average_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_rough_bergomi_realized_volatilities(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    double observations_per_year = 52.0
);

std::vector<double> generate_rough_bergomi_spot_paths(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

std::vector<double> generate_rough_bergomi_observation_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count
);

void visit_rough_bergomi_observation_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count,
    RoughBergomiObservationVisitor visitor,
    void* context
);

TimedMaxSpots generate_rough_bergomi_max_spots_timed(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
);

}  // namespace ai_factory::simulation
