#include "ai_factory/cpu/heston/common.hpp"

#include "ai_factory/cpu/common/philox.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace ai_factory::simulation {
namespace {

constexpr double kQePsiCritical = 1.5;
constexpr double kGamma1 = 0.5;
constexpr double kGamma2 = 0.5;

struct QeStep {
    double next_variance;
    double log_moment;
    bool martingale_valid;
};

struct QeCoefficients {
    double dt;
    double exp_kdt;
    double variance_linear_scale;
    double variance_constant_scale;
    double drift_dt;
    double k0;
    double k1;
    double k2;
    double k3;
    double k4;
    double martingale_a;
};

QeCoefficients make_qe_coefficients(const HestonModel& model, double dt) {
    const double xi = model.volatility_of_variance;
    const double rho = model.rho;
    const double exp_kdt = std::exp(-model.kappa * dt);
    const double one_minus_exp = 1.0 - exp_kdt;
    const double xi_squared = xi * xi;
    const double kappa_rho_over_xi = model.kappa * rho / xi;
    const double rho_over_xi = rho / xi;
    const double k2 =
        kGamma2 * dt * (kappa_rho_over_xi - 0.5) + rho_over_xi;
    const double k4 = kGamma2 * dt * (1.0 - rho * rho);
    return {
        dt,
        exp_kdt,
        xi_squared * exp_kdt * one_minus_exp / model.kappa,
        model.theta * xi_squared * one_minus_exp * one_minus_exp
            / (2.0 * model.kappa),
        (model.risk_free_rate - model.dividend_yield) * dt,
        (model.risk_free_rate - model.dividend_yield) * dt
            - rho * model.kappa * model.theta * dt / xi,
        kGamma1 * dt * (kappa_rho_over_xi - 0.5) - rho_over_xi,
        k2,
        kGamma1 * dt * (1.0 - rho * rho),
        k4,
        k2 + 0.5 * k4,
    };
}

QeStep advance_qe_variance(
    const HestonModel& model,
    double variance,
    double dt,
    double variance_normal,
    double variance_uniform,
    double martingale_a
) {
    const double kappa = model.kappa;
    const double theta = model.theta;
    const double xi = model.volatility_of_variance;
    const double exp_kdt = std::exp(-kappa * dt);
    const double one_minus_exp = 1.0 - exp_kdt;
    const double m = theta + (variance - theta) * exp_kdt;
    const double s2 =
        variance * xi * xi * exp_kdt * one_minus_exp / kappa
        + theta * xi * xi * one_minus_exp * one_minus_exp / (2.0 * kappa);

    if (m <= 0.0 || s2 <= 0.0) {
        return {0.0, 0.0, true};
    }

    const double psi = s2 / (m * m);
    if (psi <= kQePsiCritical) {
        const double inv_psi = 1.0 / psi;
        const double b2 =
            2.0 * inv_psi - 1.0
            + std::sqrt(2.0 * inv_psi)
                  * std::sqrt(std::max(2.0 * inv_psi - 1.0, 0.0));
        const double b = std::sqrt(std::max(b2, 0.0));
        const double a = m / (1.0 + b2);
        const double shifted = b + variance_normal;
        const double next_variance = a * shifted * shifted;
        const double denominator = 1.0 - 2.0 * martingale_a * a;
        if (denominator <= 0.0) {
            return {next_variance, 0.0, false};
        }
        const double log_moment =
            martingale_a * b2 * a / denominator
            - 0.5 * std::log(denominator);
        return {next_variance, log_moment, true};
    }

    const double p = (psi - 1.0) / (psi + 1.0);
    const double beta = (1.0 - p) / m;
    const double next_variance =
        variance_uniform <= p
            ? 0.0
            : std::log((1.0 - p) / (1.0 - variance_uniform)) / beta;
    if (martingale_a >= beta) {
        return {next_variance, 0.0, false};
    }
    const double moment = p + beta * (1.0 - p) / (beta - martingale_a);
    return {next_variance, std::log(moment), moment > 0.0};
}

QeStep advance_qe_variance(
    const HestonModel& model,
    const QeCoefficients& coefficients,
    double variance,
    double variance_normal,
    double variance_uniform
) {
    const double m =
        model.theta + (variance - model.theta) * coefficients.exp_kdt;
    const double s2 = variance * coefficients.variance_linear_scale
                      + coefficients.variance_constant_scale;

    if (m <= 0.0 || s2 <= 0.0) {
        return {0.0, 0.0, true};
    }

    const double psi = s2 / (m * m);
    if (psi <= kQePsiCritical) {
        const double inv_psi = 1.0 / psi;
        const double b2 =
            2.0 * inv_psi - 1.0
            + std::sqrt(2.0 * inv_psi)
                  * std::sqrt(std::max(2.0 * inv_psi - 1.0, 0.0));
        const double b = std::sqrt(std::max(b2, 0.0));
        const double a = m / (1.0 + b2);
        const double shifted = b + variance_normal;
        const double next_variance = a * shifted * shifted;
        const double denominator = 1.0 - 2.0 * coefficients.martingale_a * a;
        if (denominator <= 0.0) {
            return {next_variance, 0.0, false};
        }
        const double log_moment =
            coefficients.martingale_a * b2 * a / denominator
            - 0.5 * std::log(denominator);
        return {next_variance, log_moment, true};
    }

    const double p = (psi - 1.0) / (psi + 1.0);
    const double beta = (1.0 - p) / m;
    const double next_variance =
        variance_uniform <= p
            ? 0.0
            : std::log((1.0 - p) / (1.0 - variance_uniform)) / beta;
    if (coefficients.martingale_a >= beta) {
        return {next_variance, 0.0, false};
    }
    const double moment =
        p + beta * (1.0 - p) / (beta - coefficients.martingale_a);
    return {next_variance, std::log(moment), moment > 0.0};
}

void advance_qe_step(
    const HestonModel& model,
    HestonSimulationScheme scheme,
    const QeCoefficients& coefficients,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double& log_spot,
    double& variance
) {
    const double previous_variance = std::max(variance, 0.0);
    const auto qe = advance_qe_variance(
        model,
        coefficients,
        previous_variance,
        variance_normal,
        variance_uniform
    );
    const double next_variance = qe.next_variance;
    const double variance_integral_proxy = std::max(
        coefficients.k3 * previous_variance + coefficients.k4 * next_variance,
        0.0
    );

    if (scheme == HestonSimulationScheme::AndersenQeMartingale
        && qe.martingale_valid) {
        log_spot += coefficients.drift_dt - qe.log_moment
                    - 0.5 * coefficients.k3 * previous_variance
                    + coefficients.k2 * next_variance
                    + std::sqrt(variance_integral_proxy) * stock_normal;
    } else {
        log_spot += coefficients.k0 + coefficients.k1 * previous_variance
                    + coefficients.k2 * next_variance
                    + std::sqrt(variance_integral_proxy) * stock_normal;
    }
    variance = next_variance;
}

void advance_euler_step(
    const HestonModel& model,
    double dt,
    double sqrt_dt,
    double spot_shock,
    double independent_variance_shock,
    double& spot,
    double& variance
) {
    const double variance_floor = std::max(variance, 0.0);
    spot *= std::exp(
        (model.risk_free_rate - model.dividend_yield - 0.5 * variance_floor)
            * dt
        + std::sqrt(variance_floor) * sqrt_dt * spot_shock
    );
    const double correlation_scale = std::sqrt(1.0 - model.rho * model.rho);
    const double variance_shock =
        model.rho * spot_shock + correlation_scale * independent_variance_shock;
    variance += model.kappa * (model.theta - variance_floor) * dt;
    variance += model.volatility_of_variance * std::sqrt(variance_floor) * sqrt_dt
                * variance_shock;
    variance = std::max(variance, 0.0);
}

void advance_qe_step(
    const HestonModel& model,
    HestonSimulationScheme scheme,
    double dt,
    double variance_normal,
    double variance_uniform,
    double stock_normal,
    double& log_spot,
    double& variance
) {
    const double xi = model.volatility_of_variance;
    const double rho = model.rho;
    const double drift_dt = (model.risk_free_rate - model.dividend_yield) * dt;
    const double kappa_rho_over_xi = model.kappa * rho / xi;
    const double rho_over_xi = rho / xi;
    const double k1 = kGamma1 * dt * (kappa_rho_over_xi - 0.5) - rho_over_xi;
    const double k2 = kGamma2 * dt * (kappa_rho_over_xi - 0.5) + rho_over_xi;
    const double k3 = kGamma1 * dt * (1.0 - rho * rho);
    const double k4 = kGamma2 * dt * (1.0 - rho * rho);
    const double martingale_a = k2 + 0.5 * k4;

    const double previous_variance = std::max(variance, 0.0);
    const auto qe = advance_qe_variance(
        model,
        previous_variance,
        dt,
        variance_normal,
        variance_uniform,
        martingale_a
    );
    const double next_variance = qe.next_variance;
    const double variance_integral_proxy =
        std::max(k3 * previous_variance + k4 * next_variance, 0.0);

    if (scheme == HestonSimulationScheme::AndersenQeMartingale
        && qe.martingale_valid) {
        log_spot += drift_dt - qe.log_moment - 0.5 * k3 * previous_variance
                    + k2 * next_variance
                    + std::sqrt(variance_integral_proxy) * stock_normal;
    } else {
        const double k0 =
            drift_dt - rho * model.kappa * model.theta * dt / xi;
        log_spot += k0 + k1 * previous_variance + k2 * next_variance
                    + std::sqrt(variance_integral_proxy) * stock_normal;
    }
    variance = next_variance;
}

}  // namespace

HestonSimulationScheme parse_heston_simulation_scheme(const std::string& value) {
    if (value == "euler" || value == "euler_full_truncation") {
        return HestonSimulationScheme::EulerFullTruncation;
    }
    if (value == "qe" || value == "andersen_qe") {
        return HestonSimulationScheme::AndersenQe;
    }
    if (value == "qe_martingale" || value == "andersen_qe_martingale"
        || value == "qe-m") {
        return HestonSimulationScheme::AndersenQeMartingale;
    }
    throw std::invalid_argument("Unsupported Heston simulation scheme.");
}

std::vector<double> generate_heston_terminal_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme
) {
    if (simulation.random_backend != kPhilox4x32_10BoxMuller) {
        throw std::invalid_argument("Unsupported random backend.");
    }

    const std::size_t shock_count = simulation.num_paths * time_grid.num_steps;
    const auto first_normals = philox_standard_normals(simulation.seed, shock_count, 0);
    const auto second_normals =
        philox_standard_normals(simulation.seed, shock_count, 1);
    const auto uniforms = philox_uniforms(simulation.seed, shock_count, 2);
    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double sqrt_dt = std::sqrt(dt);

    std::vector<double> spots(simulation.num_paths, model.spot);
    std::vector<double> log_spots(
        simulation.num_paths,
        std::log(model.spot)
    );
    std::vector<double> variances(
        simulation.num_paths,
        model.initial_variance
    );

    for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            const std::size_t index = path * time_grid.num_steps + step;
            if (scheme == HestonSimulationScheme::EulerFullTruncation) {
                advance_euler_step(
                    model,
                    dt,
                    sqrt_dt,
                    first_normals[index],
                    second_normals[index],
                    spots[path],
                    variances[path]
                );
            } else {
                advance_qe_step(
                    model,
                    scheme,
                    dt,
                    first_normals[index],
                    uniforms[index],
                    second_normals[index],
                    log_spots[path],
                    variances[path]
                );
            }
        }
    }

    if (scheme != HestonSimulationScheme::EulerFullTruncation) {
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            spots[path] = std::exp(log_spots[path]);
        }
    }

    return spots;
}

std::vector<double> generate_heston_max_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme
) {
    if (simulation.random_backend != kPhilox4x32_10BoxMuller) {
        throw std::invalid_argument("Unsupported random backend.");
    }

    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const std::size_t shock_count = simulation.num_paths * time_grid.num_steps;
    const auto first_normals = philox_standard_normals(simulation.seed, shock_count, 0);
    const auto second_normals =
        philox_standard_normals(simulation.seed, shock_count, 1);
    const auto uniforms = philox_uniforms(simulation.seed, shock_count, 2);

    if (scheme != HestonSimulationScheme::EulerFullTruncation) {
        const auto coefficients = make_qe_coefficients(model, dt);
        std::vector<double> max_spots(simulation.num_paths, model.spot);
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            double log_spot = std::log(model.spot);
            double variance = model.initial_variance;
            for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
                const std::size_t index = path * time_grid.num_steps + step;
                advance_qe_step(
                    model,
                    scheme,
                    coefficients,
                    first_normals[index],
                    uniforms[index],
                    second_normals[index],
                    log_spot,
                    variance
                );
                max_spots[path] = std::max(max_spots[path], std::exp(log_spot));
            }
        }
        return max_spots;
    }

    std::vector<double> spots(simulation.num_paths, model.spot);
    std::vector<double> max_spots(simulation.num_paths, model.spot);
    std::vector<double> log_spots(
        simulation.num_paths,
        std::log(model.spot)
    );
    std::vector<double> variances(
        simulation.num_paths,
        model.initial_variance
    );

    for (std::size_t path = 0; path < simulation.num_paths; ++path) {
        for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
            const std::size_t index = path * time_grid.num_steps + step;
            advance_euler_step(
                model,
                dt,
                sqrt_dt,
                first_normals[index],
                second_normals[index],
                spots[path],
                variances[path]
            );
            max_spots[path] = std::max(max_spots[path], spots[path]);
        }
    }

    return max_spots;
}

std::vector<double> generate_heston_arithmetic_average_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme
) {
    if (simulation.random_backend != kPhilox4x32_10BoxMuller) {
        throw std::invalid_argument("Unsupported random backend.");
    }

    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const std::size_t shock_count = simulation.num_paths * time_grid.num_steps;
    const auto first_normals = philox_standard_normals(simulation.seed, shock_count, 0);
    const auto second_normals =
        philox_standard_normals(simulation.seed, shock_count, 1);
    const auto uniforms = philox_uniforms(simulation.seed, shock_count, 2);
    const double inv_step_count = 1.0 / static_cast<double>(time_grid.num_steps);

    std::vector<double> averages(simulation.num_paths, 0.0);
    if (scheme != HestonSimulationScheme::EulerFullTruncation) {
        const auto coefficients = make_qe_coefficients(model, dt);
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            double log_spot = std::log(model.spot);
            double variance = model.initial_variance;
            double sum_spot = 0.0;
            for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
                const std::size_t index = path * time_grid.num_steps + step;
                advance_qe_step(
                    model,
                    scheme,
                    coefficients,
                    first_normals[index],
                    uniforms[index],
                    second_normals[index],
                    log_spot,
                    variance
                );
                sum_spot += std::exp(log_spot);
            }
            averages[path] = sum_spot * inv_step_count;
        }
        return averages;
    }

    for (std::size_t path = 0; path < simulation.num_paths; ++path) {
        double spot = model.spot;
        double variance = model.initial_variance;
        double sum_spot = 0.0;
        for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
            const std::size_t index = path * time_grid.num_steps + step;
            advance_euler_step(
                model,
                dt,
                sqrt_dt,
                first_normals[index],
                second_normals[index],
                spot,
                variance
            );
            sum_spot += spot;
        }
        averages[path] = sum_spot * inv_step_count;
    }

    return averages;
}

std::vector<double> generate_heston_realized_volatilities(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme,
    double observations_per_year
) {
    if (simulation.random_backend != kPhilox4x32_10BoxMuller) {
        throw std::invalid_argument("Unsupported random backend.");
    }

    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const std::size_t shock_count = simulation.num_paths * time_grid.num_steps;
    const auto first_normals = philox_standard_normals(simulation.seed, shock_count, 0);
    const auto second_normals =
        philox_standard_normals(simulation.seed, shock_count, 1);
    const auto uniforms = philox_uniforms(simulation.seed, shock_count, 2);
    const double annualization =
        observations_per_year / static_cast<double>(time_grid.num_steps);

    std::vector<double> realized_volatilities(simulation.num_paths, 0.0);
    if (scheme != HestonSimulationScheme::EulerFullTruncation) {
        const auto coefficients = make_qe_coefficients(model, dt);
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            double log_spot = std::log(model.spot);
            double variance = model.initial_variance;
            double sum_squared_log_returns = 0.0;
            for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
                const double previous_log_spot = log_spot;
                const std::size_t index = path * time_grid.num_steps + step;
                advance_qe_step(
                    model,
                    scheme,
                    coefficients,
                    first_normals[index],
                    uniforms[index],
                    second_normals[index],
                    log_spot,
                    variance
                );
                const double log_return = log_spot - previous_log_spot;
                sum_squared_log_returns += log_return * log_return;
            }
            realized_volatilities[path] =
                std::sqrt(annualization * sum_squared_log_returns);
        }
        return realized_volatilities;
    }

    for (std::size_t path = 0; path < simulation.num_paths; ++path) {
        double spot = model.spot;
        double variance = model.initial_variance;
        double sum_squared_log_returns = 0.0;
        for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
            const double previous_spot = spot;
            const std::size_t index = path * time_grid.num_steps + step;
            advance_euler_step(
                model,
                dt,
                sqrt_dt,
                first_normals[index],
                second_normals[index],
                spot,
                variance
            );
            const double log_return = std::log(spot / previous_spot);
            sum_squared_log_returns += log_return * log_return;
        }
        realized_volatilities[path] =
            std::sqrt(annualization * sum_squared_log_returns);
    }

    return realized_volatilities;
}

std::vector<double> generate_heston_spot_paths(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme
) {
    if (simulation.random_backend != kPhilox4x32_10BoxMuller) {
        throw std::invalid_argument("Unsupported random backend.");
    }

    const std::size_t step_count = time_grid.num_steps + 1U;
    const std::size_t shock_count = simulation.num_paths * time_grid.num_steps;
    const auto first_normals = philox_standard_normals(simulation.seed, shock_count, 0);
    const auto second_normals =
        philox_standard_normals(simulation.seed, shock_count, 1);
    const auto uniforms = philox_uniforms(simulation.seed, shock_count, 2);
    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double sqrt_dt = std::sqrt(dt);

    std::vector<double> paths(simulation.num_paths * step_count, model.spot);
    std::vector<double> spots(simulation.num_paths, model.spot);
    std::vector<double> log_spots(
        simulation.num_paths,
        std::log(model.spot)
    );
    std::vector<double> variances(
        simulation.num_paths,
        model.initial_variance
    );

    for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            const std::size_t index = path * time_grid.num_steps + step;
            const std::size_t output_index = path * step_count + step + 1U;
            if (scheme == HestonSimulationScheme::EulerFullTruncation) {
                advance_euler_step(
                    model,
                    dt,
                    sqrt_dt,
                    first_normals[index],
                    second_normals[index],
                    spots[path],
                    variances[path]
                );
                paths[output_index] = spots[path];
            } else {
                advance_qe_step(
                    model,
                    scheme,
                    dt,
                    first_normals[index],
                    uniforms[index],
                    second_normals[index],
                    log_spots[path],
                    variances[path]
                );
                paths[output_index] = std::exp(log_spots[path]);
            }
        }
    }

    return paths;
}

HestonStatePaths generate_heston_state_paths(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme
) {
    if (simulation.random_backend != kPhilox4x32_10BoxMuller) {
        throw std::invalid_argument("Unsupported random backend.");
    }

    const std::size_t step_count = time_grid.num_steps + 1U;
    const std::size_t shock_count = simulation.num_paths * time_grid.num_steps;
    const auto first_normals = philox_standard_normals(
        simulation.seed, shock_count, 0
    );
    const auto second_normals = philox_standard_normals(
        simulation.seed, shock_count, 1
    );
    const auto uniforms = philox_uniforms(simulation.seed, shock_count, 2);
    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double sqrt_dt = std::sqrt(dt);

    HestonStatePaths result{
        std::vector<double>(simulation.num_paths * step_count, model.spot),
        std::vector<double>(
            simulation.num_paths * step_count, model.initial_variance
        ),
    };
    std::vector<double> spots(simulation.num_paths, model.spot);
    std::vector<double> log_spots(simulation.num_paths, std::log(model.spot));
    std::vector<double> variances(simulation.num_paths, model.initial_variance);

    for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
        for (std::size_t path = 0; path < simulation.num_paths; ++path) {
            const std::size_t index = path * time_grid.num_steps + step;
            const std::size_t output_index = path * step_count + step + 1U;
            if (scheme == HestonSimulationScheme::EulerFullTruncation) {
                advance_euler_step(
                    model, dt, sqrt_dt,
                    first_normals[index], second_normals[index],
                    spots[path], variances[path]
                );
                result.spots[output_index] = spots[path];
            } else {
                advance_qe_step(
                    model, scheme, dt,
                    first_normals[index], uniforms[index], second_normals[index],
                    log_spots[path], variances[path]
                );
                result.spots[output_index] = std::exp(log_spots[path]);
            }
            result.variances[output_index] = variances[path];
        }
    }
    return result;
}

std::vector<double> generate_heston_observation_spots(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    std::size_t observation_count,
    HestonSimulationScheme scheme
) {
    if (simulation.random_backend != kPhilox4x32_10BoxMuller) {
        throw std::invalid_argument("Unsupported random backend.");
    }
    if (observation_count == 0U
        || time_grid.num_steps % observation_count != 0U) {
        throw std::invalid_argument(
            "Observation count must divide the Heston step count."
        );
    }
    const auto shock_count = simulation.num_paths * time_grid.num_steps;
    const auto first_normals = philox_standard_normals(
        simulation.seed, shock_count, 0
    );
    const auto second_normals = philox_standard_normals(
        simulation.seed, shock_count, 1
    );
    const auto uniforms = philox_uniforms(simulation.seed, shock_count, 2);
    const double dt = time_grid.maturity / static_cast<double>(time_grid.num_steps);
    const double sqrt_dt = std::sqrt(dt);
    const auto stride = time_grid.num_steps / observation_count;
    const auto coefficients = make_qe_coefficients(model, dt);
    std::vector<double> observations(simulation.num_paths * observation_count);
    for (std::size_t path = 0; path < simulation.num_paths; ++path) {
        double spot = model.spot;
        double log_spot = std::log(model.spot);
        double variance = model.initial_variance;
        std::size_t observation = 0U;
        for (std::size_t step = 0; step < time_grid.num_steps; ++step) {
            const auto shock_index = path * time_grid.num_steps + step;
            if (scheme == HestonSimulationScheme::EulerFullTruncation) {
                advance_euler_step(
                    model,
                    dt,
                    sqrt_dt,
                    first_normals[shock_index],
                    second_normals[shock_index],
                    spot,
                    variance
                );
            } else {
                advance_qe_step(
                    model,
                    scheme,
                    coefficients,
                    first_normals[shock_index],
                    uniforms[shock_index],
                    second_normals[shock_index],
                    log_spot,
                    variance
                );
                spot = std::exp(log_spot);
            }
            if ((step + 1U) % stride == 0U) {
                observations[path * observation_count + observation] = spot;
                ++observation;
            }
        }
    }
    return observations;
}

TimedTerminalSpots generate_heston_terminal_spots_timed(
    const HestonModel& model,
    const TimeGrid& time_grid,
    const SimulationConfig& simulation,
    HestonSimulationScheme scheme
) {
    const auto start = std::chrono::steady_clock::now();
    auto terminal_spots = generate_heston_terminal_spots(
        model,
        time_grid,
        simulation,
        scheme
    );
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    return {std::move(terminal_spots), elapsed.count()};
}

}  // namespace ai_factory::simulation
