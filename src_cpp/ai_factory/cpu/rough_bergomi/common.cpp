#include "ai_factory/cpu/rough_bergomi/common.hpp"

#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ai_factory::simulation {
namespace {

double optimal_hybrid_evaluation_point(double alpha, std::size_t k) {
    const double kd = static_cast<double>(k);
    const double average =
        (std::pow(kd, alpha + 1.0) - std::pow(kd - 1.0, alpha + 1.0))
        / (alpha + 1.0);
    return std::pow(average, 1.0 / alpha);
}

void validate_model(const RoughBergomiModel& model) {
    if (model.alpha <= -0.5 || model.alpha >= 0.0) {
        throw std::invalid_argument("Rough Bergomi alpha must be in (-0.5, 0).");
    }
    if (model.forward_variance <= 0.0 || model.eta <= 0.0) {
        throw std::invalid_argument(
            "Rough Bergomi forward variance and eta must be positive."
        );
    }
    if (model.rho <= -1.0 || model.rho >= 1.0) {
        throw std::invalid_argument("Rough Bergomi rho must be in (-1, 1).");
    }
}

}  // namespace

static std::vector<double> generate_rough_bergomi_terminal_statistic(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    bool track_maximum
) {
    validate_model(model);

    const auto num_paths = simulation.num_paths;
    const auto num_steps = time_grid.num_steps;
    const double maturity = time_grid.maturity;
    const double dt = maturity / static_cast<double>(num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double drift = (model.risk_free_rate - model.dividend_yield) * dt;
    const double rho_perp = std::sqrt(1.0 - model.rho * model.rho);
    const double alpha = model.alpha;
    const double variance_scale = std::sqrt(2.0 * alpha + 1.0);
    const double singular_covariance_scale =
        std::pow(dt, alpha + 0.5) / (alpha + 1.0);
    const double singular_residual_variance =
        std::pow(dt, 2.0 * alpha + 1.0)
        * (1.0 / (2.0 * alpha + 1.0)
           - 1.0 / ((alpha + 1.0) * (alpha + 1.0)));
    const double singular_residual_scale =
        std::sqrt(std::max(singular_residual_variance, 0.0));

    std::vector<double> weights(num_steps + 1U, 0.0);
    for (std::size_t k = 2U; k <= num_steps; ++k) {
        weights[k] = std::pow(
            optimal_hybrid_evaluation_point(alpha, k) * dt,
            alpha
        );
    }

    const auto w_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 0);
    const auto singular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 1);
    const auto perpendicular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 2);

    std::vector<double> max_spots(num_paths);
    #pragma omp parallel
    {
    std::vector<double> dws(num_steps);
    std::vector<double> singular_terms(num_steps);
    #pragma omp for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        for (std::size_t step = 0; step < num_steps; ++step) {
            const auto index = path * num_steps + step;
            const double w_normal = w_normals[index];
            dws[step] = sqrt_dt * w_normal;
            singular_terms[step] =
                singular_covariance_scale * w_normal
                + singular_residual_scale * singular_normals[index];
        }

        double log_spot = std::log(model.spot);
        double max_spot = model.spot;
        for (std::size_t step = 0; step < num_steps; ++step) {
            double y = 0.0;
            if (step > 0U) {
                y += singular_terms[step - 1U];
                for (std::size_t k = 2U; k <= step; ++k) {
                    y += weights[k] * dws[step - k];
                }
            }
            const double time = static_cast<double>(step) * dt;
            const double variance =
                model.forward_variance
                * std::exp(
                    model.eta * variance_scale * y
                    - 0.5 * model.eta * model.eta
                          * std::pow(time, 2.0 * alpha + 1.0)
                );
            const auto index = path * num_steps + step;
            const double dz =
                model.rho * dws[step]
                + rho_perp * sqrt_dt * perpendicular_normals[index];
            log_spot += drift - 0.5 * variance * dt + std::sqrt(variance) * dz;
            max_spot = std::max(max_spot, std::exp(log_spot));
        }
        max_spots[path] = track_maximum ? max_spot : std::exp(log_spot);
    }
    }

    return max_spots;
}

std::vector<double> generate_rough_bergomi_max_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    return generate_rough_bergomi_terminal_statistic(
        model, time_grid, simulation, true
    );
}

std::vector<double> generate_rough_bergomi_terminal_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    return generate_rough_bergomi_terminal_statistic(
        model, time_grid, simulation, false
    );
}

std::vector<double> generate_rough_bergomi_arithmetic_average_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    validate_model(model);

    const auto num_paths = simulation.num_paths;
    const auto num_steps = time_grid.num_steps;
    const double maturity = time_grid.maturity;
    const double dt = maturity / static_cast<double>(num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double drift = (model.risk_free_rate - model.dividend_yield) * dt;
    const double rho_perp = std::sqrt(1.0 - model.rho * model.rho);
    const double alpha = model.alpha;
    const double variance_scale = std::sqrt(2.0 * alpha + 1.0);
    const double singular_covariance_scale =
        std::pow(dt, alpha + 0.5) / (alpha + 1.0);
    const double singular_residual_variance =
        std::pow(dt, 2.0 * alpha + 1.0)
        * (1.0 / (2.0 * alpha + 1.0)
           - 1.0 / ((alpha + 1.0) * (alpha + 1.0)));
    const double singular_residual_scale =
        std::sqrt(std::max(singular_residual_variance, 0.0));

    std::vector<double> weights(num_steps + 1U, 0.0);
    for (std::size_t k = 2U; k <= num_steps; ++k) {
        weights[k] = std::pow(
            optimal_hybrid_evaluation_point(alpha, k) * dt,
            alpha
        );
    }

    const auto w_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 0);
    const auto singular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 1);
    const auto perpendicular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 2);

    std::vector<double> average_spots(num_paths);
    const double inv_steps = 1.0 / static_cast<double>(num_steps);
    #pragma omp parallel
    {
    std::vector<double> dws(num_steps);
    std::vector<double> singular_terms(num_steps);
    #pragma omp for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        for (std::size_t step = 0; step < num_steps; ++step) {
            const auto index = path * num_steps + step;
            const double w_normal = w_normals[index];
            dws[step] = sqrt_dt * w_normal;
            singular_terms[step] =
                singular_covariance_scale * w_normal
                + singular_residual_scale * singular_normals[index];
        }

        double log_spot = std::log(model.spot);
        double sum_spot = 0.0;
        for (std::size_t step = 0; step < num_steps; ++step) {
            double y = 0.0;
            if (step > 0U) {
                y += singular_terms[step - 1U];
                for (std::size_t k = 2U; k <= step; ++k) {
                    y += weights[k] * dws[step - k];
                }
            }
            const double time = static_cast<double>(step) * dt;
            const double variance =
                model.forward_variance
                * std::exp(
                    model.eta * variance_scale * y
                    - 0.5 * model.eta * model.eta
                          * std::pow(time, 2.0 * alpha + 1.0)
                );
            const auto index = path * num_steps + step;
            const double dz =
                model.rho * dws[step]
                + rho_perp * sqrt_dt * perpendicular_normals[index];
            log_spot += drift - 0.5 * variance * dt + std::sqrt(variance) * dz;
            sum_spot += std::exp(log_spot);
        }
        average_spots[path] = sum_spot * inv_steps;
    }
    }

    return average_spots;
}

std::vector<double> generate_rough_bergomi_realized_volatilities(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    double observations_per_year
) {
    validate_model(model);

    const auto num_paths = simulation.num_paths;
    const auto num_steps = time_grid.num_steps;
    const double maturity = time_grid.maturity;
    const double dt = maturity / static_cast<double>(num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double drift = (model.risk_free_rate - model.dividend_yield) * dt;
    const double rho_perp = std::sqrt(1.0 - model.rho * model.rho);
    const double alpha = model.alpha;
    const double variance_scale = std::sqrt(2.0 * alpha + 1.0);
    const double singular_covariance_scale =
        std::pow(dt, alpha + 0.5) / (alpha + 1.0);
    const double singular_residual_variance =
        std::pow(dt, 2.0 * alpha + 1.0)
        * (1.0 / (2.0 * alpha + 1.0)
           - 1.0 / ((alpha + 1.0) * (alpha + 1.0)));
    const double singular_residual_scale =
        std::sqrt(std::max(singular_residual_variance, 0.0));

    std::vector<double> weights(num_steps + 1U, 0.0);
    for (std::size_t k = 2U; k <= num_steps; ++k) {
        weights[k] = std::pow(
            optimal_hybrid_evaluation_point(alpha, k) * dt,
            alpha
        );
    }

    const auto w_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 0);
    const auto singular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 1);
    const auto perpendicular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 2);

    std::vector<double> realized_volatilities(num_paths);
    const double annualization =
        observations_per_year / static_cast<double>(num_steps);
    #pragma omp parallel
    {
    std::vector<double> dws(num_steps);
    std::vector<double> singular_terms(num_steps);
    #pragma omp for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        for (std::size_t step = 0; step < num_steps; ++step) {
            const auto index = path * num_steps + step;
            const double w_normal = w_normals[index];
            dws[step] = sqrt_dt * w_normal;
            singular_terms[step] =
                singular_covariance_scale * w_normal
                + singular_residual_scale * singular_normals[index];
        }

        double sum_squared_log_returns = 0.0;
        for (std::size_t step = 0; step < num_steps; ++step) {
            double y = 0.0;
            if (step > 0U) {
                y += singular_terms[step - 1U];
                for (std::size_t k = 2U; k <= step; ++k) {
                    y += weights[k] * dws[step - k];
                }
            }
            const double time = static_cast<double>(step) * dt;
            const double variance =
                model.forward_variance
                * std::exp(
                    model.eta * variance_scale * y
                    - 0.5 * model.eta * model.eta
                          * std::pow(time, 2.0 * alpha + 1.0)
                );
            const auto index = path * num_steps + step;
            const double dz =
                model.rho * dws[step]
                + rho_perp * sqrt_dt * perpendicular_normals[index];
            const double log_return =
                drift - 0.5 * variance * dt + std::sqrt(variance) * dz;
            sum_squared_log_returns += log_return * log_return;
        }
        realized_volatilities[path] =
            std::sqrt(annualization * sum_squared_log_returns);
    }
    }

    return realized_volatilities;
}

std::vector<double> generate_rough_bergomi_spot_paths(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    validate_model(model);
    const auto num_paths = simulation.num_paths;
    const auto num_steps = time_grid.num_steps;
    const double dt = time_grid.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double drift = (model.risk_free_rate - model.dividend_yield) * dt;
    const double rho_perp = std::sqrt(1.0 - model.rho * model.rho);
    const double variance_scale = std::sqrt(2.0 * model.alpha + 1.0);
    std::vector<double> weights(num_steps + 1U, 0.0);
    for (std::size_t k = 2U; k <= num_steps; ++k) {
        weights[k] = std::pow(
            optimal_hybrid_evaluation_point(model.alpha, k) * dt,
            model.alpha
        );
    }
    const double singular_covariance_scale =
        std::pow(dt, model.alpha + 0.5) / (model.alpha + 1.0);
    const double singular_residual_variance =
        std::pow(dt, 2.0 * model.alpha + 1.0)
        * (1.0 / (2.0 * model.alpha + 1.0)
           - 1.0 / ((model.alpha + 1.0) * (model.alpha + 1.0)));
    const double singular_residual_scale =
        std::sqrt(std::max(singular_residual_variance, 0.0));
    const auto path_width = num_steps + 1U;
    std::vector<double> spot_paths(num_paths * path_width);
    const auto w_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 0);
    const auto singular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 1);
    const auto perpendicular_normals =
        philox_standard_normals(simulation.seed, num_paths * num_steps, 2);

    #pragma omp parallel
    {
    std::vector<double> dws(num_steps);
    std::vector<double> singular_terms(num_steps);
    #pragma omp for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        const auto base = path * path_width;
        for (std::size_t step = 0; step < num_steps; ++step) {
            const auto index = path * num_steps + step;
            const double w_normal = w_normals[index];
            dws[step] = sqrt_dt * w_normal;
            singular_terms[step] =
                singular_covariance_scale * w_normal
                + singular_residual_scale * singular_normals[index];
        }

        double log_spot = std::log(model.spot);
        spot_paths[base] = model.spot;
        for (std::size_t step = 0; step < num_steps; ++step) {
            double y = 0.0;
            if (step > 0U) {
                y += singular_terms[step - 1U];
                for (std::size_t k = 2U; k <= step; ++k) {
                    y += weights[k] * dws[step - k];
                }
            }
            const double time = static_cast<double>(step) * dt;
            const double variance =
                model.forward_variance
                * std::exp(
                    model.eta * variance_scale * y
                    - 0.5 * model.eta * model.eta
                          * std::pow(time, 2.0 * model.alpha + 1.0)
                );
            const auto normal_index = path * num_steps + step;
            const double dz =
                model.rho * dws[step]
                + rho_perp * sqrt_dt * perpendicular_normals[normal_index];
            log_spot += drift - 0.5 * variance * dt + std::sqrt(variance) * dz;
            spot_paths[base + step + 1U] = std::exp(log_spot);
        }
    }
    }

    return spot_paths;
}

void visit_rough_bergomi_observation_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count,
    RoughBergomiObservationVisitor visitor,
    void* context
) {
    validate_model(model);
    if (visitor == nullptr) {
        throw std::invalid_argument("Rough Bergomi observation visitor is null.");
    }
    if (observation_count == 0U
        || time_grid.num_steps % observation_count != 0U) {
        throw std::invalid_argument(
            "Observation count must divide the Rough Bergomi step count."
        );
    }
    const auto num_paths = simulation.num_paths;
    const auto num_steps = time_grid.num_steps;
    const double dt = time_grid.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const double drift = (model.risk_free_rate - model.dividend_yield) * dt;
    const double rho_perp = std::sqrt(1.0 - model.rho * model.rho);
    const double variance_scale = std::sqrt(2.0 * model.alpha + 1.0);
    std::vector<double> weights(num_steps + 1U, 0.0);
    for (std::size_t k = 2U; k <= num_steps; ++k) {
        weights[k] = std::pow(
            optimal_hybrid_evaluation_point(model.alpha, k) * dt,
            model.alpha
        );
    }
    const double singular_covariance_scale =
        std::pow(dt, model.alpha + 0.5) / (model.alpha + 1.0);
    const double singular_residual_variance =
        std::pow(dt, 2.0 * model.alpha + 1.0)
        * (1.0 / (2.0 * model.alpha + 1.0)
           - 1.0 / ((model.alpha + 1.0) * (model.alpha + 1.0)));
    const double singular_residual_scale =
        std::sqrt(std::max(singular_residual_variance, 0.0));
    const auto w_normals = philox_standard_normals(
        simulation.seed, num_paths * num_steps, 0
    );
    const auto singular_normals = philox_standard_normals(
        simulation.seed, num_paths * num_steps, 1
    );
    const auto perpendicular_normals = philox_standard_normals(
        simulation.seed, num_paths * num_steps, 2
    );
    const auto stride = num_steps / observation_count;
    #pragma omp parallel
    {
    std::vector<double> dws(num_steps);
    std::vector<double> singular_terms(num_steps);
    #pragma omp for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        for (std::size_t step = 0; step < num_steps; ++step) {
            const auto shock_index = path * num_steps + step;
            const double w_normal = w_normals[shock_index];
            dws[step] = sqrt_dt * w_normal;
            singular_terms[step] =
                singular_covariance_scale * w_normal
                + singular_residual_scale * singular_normals[shock_index];
        }
        double log_spot = std::log(model.spot);
        std::size_t observation = 0U;
        for (std::size_t step = 0; step < num_steps; ++step) {
            double y = 0.0;
            if (step > 0U) {
                y += singular_terms[step - 1U];
                for (std::size_t k = 2U; k <= step; ++k) {
                    y += weights[k] * dws[step - k];
                }
            }
            const double time = static_cast<double>(step) * dt;
            const double variance = model.forward_variance
                * std::exp(
                    model.eta * variance_scale * y
                    - 0.5 * model.eta * model.eta
                          * std::pow(time, 2.0 * model.alpha + 1.0)
                );
            const auto shock_index = path * num_steps + step;
            const double dz =
                model.rho * dws[step]
                + rho_perp * sqrt_dt * perpendicular_normals[shock_index];
            log_spot += drift - 0.5 * variance * dt + std::sqrt(variance) * dz;
            if ((step + 1U) % stride == 0U) {
                ++observation;
                if (visitor(path, observation, std::exp(log_spot), context)) {
                    break;
                }
            }
        }
    }
    }
}

std::vector<double> generate_rough_bergomi_observation_spots(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count
) {
    std::vector<double> observations(
        simulation.num_paths * observation_count
    );
    struct StorageContext {
        double* values;
        std::size_t observation_count;
    } context{observations.data(), observation_count};
    const auto store_observation = [](
        std::size_t path,
        std::size_t observation,
        double spot,
        void* raw_context
    ) -> bool {
        auto& storage = *static_cast<StorageContext*>(raw_context);
        storage.values[
            path * storage.observation_count + observation - 1U
        ] = spot;
        return false;
    };
    visit_rough_bergomi_observation_spots(
        model,
        time_grid,
        simulation,
        observation_count,
        store_observation,
        &context
    );
    return observations;
}

TimedMaxSpots generate_rough_bergomi_max_spots_timed(
    const RoughBergomiModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation
) {
    const auto start = std::chrono::steady_clock::now();
    auto max_spots = generate_rough_bergomi_max_spots(model, time_grid, simulation);
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    return {std::move(max_spots), elapsed.count()};
}

}  // namespace ai_factory::simulation
