#include "ai_factory/cpu/black_scholes/common.hpp"

#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ai_factory::simulation {
namespace {

struct PathSummary {
    double terminal = 0.0;
    double maximum = 0.0;
    double arithmetic_average = 0.0;
    double realized_volatility = 0.0;
};

PathSummary simulate_path_summary(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const std::vector<double>& normals,
    std::size_t path,
    std::size_t num_steps
) {
    const double dt = time_grid.maturity / static_cast<double>(num_steps);
    const double drift =
        (model.risk_free_rate - model.dividend_yield
         - 0.5 * model.volatility * model.volatility)
        * dt;
    const double diffusion = model.volatility * std::sqrt(dt);
    double spot = model.spot;
    double maximum = model.spot;
    double average_sum = 0.0;
    double realized_variance_sum = 0.0;
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double log_return = drift + diffusion * normals[path * num_steps + step];
        spot *= std::exp(log_return);
        maximum = std::max(maximum, spot);
        average_sum += spot;
        realized_variance_sum += log_return * log_return;
    }
    return {
        spot,
        maximum,
        average_sum / static_cast<double>(num_steps),
        std::sqrt(realized_variance_sum / time_grid.maturity),
    };
}

std::vector<double> generate_statistic(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    double (*selector)(const PathSummary&)
) {
    const auto normals = philox_standard_normals(
        simulation.seed,
        simulation.num_paths * time_grid.num_steps,
        0
    );
    std::vector<double> values(simulation.num_paths);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(simulation.num_paths);
         ++index) {
        const auto path = static_cast<std::size_t>(index);
        values[path] = selector(
            simulate_path_summary(model, time_grid, normals, path, time_grid.num_steps)
        );
    }
    return values;
}

double select_maximum(const PathSummary& summary) {
    return summary.maximum;
}

double select_average(const PathSummary& summary) {
    return summary.arithmetic_average;
}

double select_realized_volatility(const PathSummary& summary) {
    return summary.realized_volatility;
}

}  // namespace

std::vector<double> generate_black_scholes_max_spots(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    return generate_statistic(model, time_grid, simulation, select_maximum);
}

std::vector<double> generate_black_scholes_arithmetic_average_spots(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    return generate_statistic(model, time_grid, simulation, select_average);
}

std::vector<double> generate_black_scholes_realized_volatilities(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    return generate_statistic(model, time_grid, simulation, select_realized_volatility);
}

std::vector<double> generate_black_scholes_spot_paths(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    const auto normals = philox_standard_normals(
        simulation.seed,
        simulation.num_paths * time_grid.num_steps,
        0
    );
    std::vector<double> paths(simulation.num_paths * (time_grid.num_steps + 1U));
    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double drift =
        (model.risk_free_rate - model.dividend_yield
         - 0.5 * model.volatility * model.volatility)
        * dt;
    const double diffusion = model.volatility * std::sqrt(dt);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0;
         index < static_cast<std::ptrdiff_t>(simulation.num_paths);
         ++index) {
        const auto path = static_cast<std::size_t>(index);
        double spot = model.spot;
        paths[path * (time_grid.num_steps + 1U)] = spot;
        for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
            spot *= std::exp(drift + diffusion * normals[path * time_grid.num_steps + step]);
            paths[path * (time_grid.num_steps + 1U) + step + 1U] = spot;
        }
    }
    return paths;
}

std::vector<double> generate_black_scholes_observation_spots(
    const BlackScholesModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count
) {
    if (observation_count == 0U
        || time_grid.num_steps % observation_count != 0U) {
        throw std::invalid_argument(
            "Observation count must divide the Black-Scholes step count."
        );
    }
    const auto normals = philox_standard_normals(
        simulation.seed,
        simulation.num_paths * time_grid.num_steps,
        0
    );
    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double drift =
        (model.risk_free_rate - model.dividend_yield
         - 0.5 * model.volatility * model.volatility)
        * dt;
    const double diffusion = model.volatility * std::sqrt(dt);
    const auto stride = time_grid.num_steps / observation_count;
    std::vector<double> observations(simulation.num_paths * observation_count);
    for (std::size_t path = 0; path < simulation.num_paths; ++path) {
        double spot = model.spot;
        std::size_t observation = 0U;
        for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
            spot *= std::exp(
                drift + diffusion * normals[path * time_grid.num_steps + step]
            );
            if ((step + 1U) % stride == 0U) {
                observations[path * observation_count + observation] = spot;
                ++observation;
            }
        }
    }
    return observations;
}

}  // namespace ai_factory::simulation
