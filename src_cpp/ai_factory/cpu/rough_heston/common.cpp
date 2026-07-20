#include "ai_factory/cpu/rough_heston/common.hpp"

#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cuda/common/types.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ai_factory::simulation {
namespace {

constexpr std::size_t kFactors = cuda::kRoughHestonFactorCount;

struct Coefficients {
    std::array<double, kFactors> decay{};
    std::array<double, kFactors> drift{};
    std::array<double, kFactors> diffusion{};
};

void validate(const RoughHestonModel& model) {
    if (!(model.hurst > 0.0 && model.hurst < 0.5)) {
        throw std::invalid_argument("Rough Heston Hurst must be in (0, 0.5).");
    }
    if (model.initial_variance <= 0.0 || model.theta <= 0.0
        || model.kappa <= 0.0 || model.volatility_of_variance <= 0.0) {
        throw std::invalid_argument("Rough Heston variance parameters must be positive.");
    }
    if (!(model.rho > -1.0 && model.rho < 1.0)) {
        throw std::invalid_argument("Rough Heston rho must be in (-1, 1).");
    }
}

Coefficients coefficients(double hurst, double maturity, double dt) {
    Coefficients result;
    const double alpha = hurst + 0.5;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double measure_scale = std::sin(pi * alpha) / pi;
    const double lower_scale = 0.1 / maturity;
    const double upper_scale = 20.0 / dt;
    const double ratio = std::pow(
        upper_scale / lower_scale,
        1.0 / static_cast<double>(kFactors - 1U)
    );
    double left = 0.0;
    double right = lower_scale;
    for (std::size_t factor = 0; factor < kFactors; ++factor) {
        const double mass = measure_scale
            * (std::pow(right, 1.0 - alpha) - std::pow(left, 1.0 - alpha))
            / (1.0 - alpha);
        const double first_moment = measure_scale
            * (std::pow(right, 2.0 - alpha) - std::pow(left, 2.0 - alpha))
            / (2.0 - alpha);
        const double node = first_moment / mass;
        const double integrated = mass * -std::expm1(-node * dt) / node;
        result.decay[factor] = std::exp(-node * dt);
        result.drift[factor] = integrated;
        result.diffusion[factor] = integrated / std::sqrt(dt);
        left = right;
        right *= ratio;
    }
    return result;
}

template <typename Observer>
double simulate_path(
    const RoughHestonModel& model,
    const Coefficients& coefficient,
    std::size_t path,
    std::size_t num_steps,
    double dt,
    const std::vector<double>& variance_normals,
    const std::vector<double>& independent_normals,
    Observer&& observer
) {
    std::array<double, kFactors> factors{};
    double variance = model.initial_variance;
    double log_spot = std::log(model.spot);
    const double rho_perp = std::sqrt(1.0 - model.rho * model.rho);
    const double sqrt_dt = std::sqrt(dt);
    for (std::size_t step = 0; step < num_steps; ++step) {
        const auto index = path * num_steps + step;
        const double positive_variance = std::max(variance, 0.0);
        const double root_variance = std::sqrt(positive_variance);
        const double z_variance = variance_normals[index];
        const double common_drift = model.kappa * (model.theta - positive_variance);
        const double common_diffusion =
            model.volatility_of_variance * root_variance * z_variance;
        double factor_sum = 0.0;
        for (std::size_t factor = 0; factor < kFactors; ++factor) {
            factors[factor] = coefficient.decay[factor] * factors[factor]
                + coefficient.drift[factor] * common_drift
                + coefficient.diffusion[factor] * common_diffusion;
            factor_sum += factors[factor];
        }
        const double stock_normal =
            model.rho * z_variance + rho_perp * independent_normals[index];
        const double log_return =
            (model.risk_free_rate - model.dividend_yield
             - 0.5 * positive_variance) * dt
            + root_variance * sqrt_dt * stock_normal;
        log_spot += log_return;
        variance = std::max(model.initial_variance + factor_sum, 0.0);
        observer(step, std::exp(log_spot), log_return, variance, factors);
    }
    return std::exp(log_spot);
}

template <typename Factory>
std::vector<double> generate_statistic(
    const RoughHestonModel& model,
    const TimeGrid& grid,
    const SimulationConfig& simulation,
    Factory&& factory
) {
    validate(model);
    const double dt = grid.maturity / static_cast<double>(grid.num_steps);
    const auto coefficient = coefficients(model.hurst, grid.maturity, dt);
    const auto count = simulation.num_paths * grid.num_steps;
    const auto variance_normals = philox_standard_normals(simulation.seed, count, 0U);
    const auto independent_normals = philox_standard_normals(simulation.seed, count, 1U);
    std::vector<double> values(simulation.num_paths);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(simulation.num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        auto state = factory.initial(model, grid);
        const double terminal = simulate_path(
            model, coefficient, path, grid.num_steps, dt,
            variance_normals, independent_normals,
            [&] (
                std::size_t step, double spot, double log_return,
                double variance, const std::array<double, kFactors>&
            ) {
                factory.observe(state, step, spot, log_return, variance);
            }
        );
        values[path] = factory.finish(state, terminal, grid);
    }
    return values;
}

struct TerminalFactory {
    struct State {};
    State initial(const RoughHestonModel&, const TimeGrid&) const { return {}; }
    void observe(State&, std::size_t, double, double, double) const {}
    double finish(State&, double terminal, const TimeGrid&) const { return terminal; }
};

struct MaximumFactory {
    struct State { double value; };
    State initial(const RoughHestonModel& model, const TimeGrid&) const { return {model.spot}; }
    void observe(State& state, std::size_t, double spot, double, double) const {
        state.value = std::max(state.value, spot);
    }
    double finish(State& state, double, const TimeGrid&) const { return state.value; }
};

struct AverageFactory {
    struct State { double sum; };
    State initial(const RoughHestonModel&, const TimeGrid&) const { return {0.0}; }
    void observe(State& state, std::size_t, double spot, double, double) const { state.sum += spot; }
    double finish(State& state, double, const TimeGrid& grid) const {
        return state.sum / static_cast<double>(grid.num_steps);
    }
};

struct RealizedVolatilityFactory {
    double observations_per_year;
    struct State { double sumsq; };
    State initial(const RoughHestonModel&, const TimeGrid&) const { return {0.0}; }
    void observe(State& state, std::size_t, double, double log_return, double) const {
        state.sumsq += log_return * log_return;
    }
    double finish(State& state, double, const TimeGrid& grid) const {
        return std::sqrt(
            observations_per_year / static_cast<double>(grid.num_steps) * state.sumsq
        );
    }
};

}  // namespace

std::vector<double> generate_rough_heston_terminal_spots(
    const RoughHestonModel& m, const TimeGrid& g, const SimulationConfig& s
) { return generate_statistic(m, g, s, TerminalFactory{}); }

std::vector<double> generate_rough_heston_max_spots(
    const RoughHestonModel& m, const TimeGrid& g, const SimulationConfig& s
) { return generate_statistic(m, g, s, MaximumFactory{}); }

std::vector<double> generate_rough_heston_arithmetic_average_spots(
    const RoughHestonModel& m, const TimeGrid& g, const SimulationConfig& s
) { return generate_statistic(m, g, s, AverageFactory{}); }

std::vector<double> generate_rough_heston_realized_volatilities(
    const RoughHestonModel& m, const TimeGrid& g, const SimulationConfig& s,
    double observations_per_year
) { return generate_statistic(m, g, s, RealizedVolatilityFactory{observations_per_year}); }

std::vector<double> generate_rough_heston_spot_paths(
    const RoughHestonModel& model, const TimeGrid& grid,
    const SimulationConfig& simulation
) {
    validate(model);
    const double dt = grid.maturity / static_cast<double>(grid.num_steps);
    const auto coefficient = coefficients(model.hurst, grid.maturity, dt);
    const auto count = simulation.num_paths * grid.num_steps;
    const auto variance_normals = philox_standard_normals(simulation.seed, count, 0U);
    const auto independent_normals = philox_standard_normals(simulation.seed, count, 1U);
    const auto width = grid.num_steps + 1U;
    std::vector<double> paths(simulation.num_paths * width);
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(simulation.num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        paths[path * width] = model.spot;
        simulate_path(
            model, coefficient, path, grid.num_steps, dt,
            variance_normals, independent_normals,
            [&] (
                std::size_t step, double spot, double, double,
                const std::array<double, kFactors>&
            ) {
                paths[path * width + step + 1U] = spot;
            }
        );
    }
    return paths;
}

RoughHestonStatePaths generate_rough_heston_state_paths(
    const RoughHestonModel& model, const TimeGrid& grid,
    const SimulationConfig& simulation
) {
    validate(model);
    const double dt = grid.maturity / static_cast<double>(grid.num_steps);
    const auto coefficient = coefficients(model.hurst, grid.maturity, dt);
    const auto count = simulation.num_paths * grid.num_steps;
    const auto variance_normals = philox_standard_normals(
        simulation.seed, count, 0U
    );
    const auto independent_normals = philox_standard_normals(
        simulation.seed, count, 1U
    );
    const auto width = grid.num_steps + 1U;
    RoughHestonStatePaths result{
        std::vector<double>(simulation.num_paths * width),
        std::vector<double>(simulation.num_paths * width * kFactors, 0.0),
    };
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(simulation.num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        result.spots[path * width] = model.spot;
        simulate_path(
            model, coefficient, path, grid.num_steps, dt,
            variance_normals, independent_normals,
            [&] (
                std::size_t step, double spot, double, double,
                const std::array<double, kFactors>& factors
            ) {
                const auto date = step + 1U;
                result.spots[path * width + date] = spot;
                const auto offset = (path * width + date) * kFactors;
                std::copy(factors.begin(), factors.end(), result.factors.begin() + offset);
            }
        );
    }
    return result;
}

void visit_rough_heston_observation_spots(
    const RoughHestonModel& model, const TimeGrid& grid,
    const SimulationConfig& simulation, std::size_t observation_count,
    RoughHestonObservationVisitor visitor, void* context
) {
    if (observation_count == 0U || grid.num_steps % observation_count != 0U) {
        throw std::invalid_argument("Observation count must divide Rough Heston steps.");
    }
    const auto paths = generate_rough_heston_spot_paths(model, grid, simulation);
    const auto width = grid.num_steps + 1U;
    const auto stride = grid.num_steps / observation_count;
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t signed_path = 0;
         signed_path < static_cast<std::ptrdiff_t>(simulation.num_paths);
         ++signed_path) {
        const auto path = static_cast<std::size_t>(signed_path);
        for (std::size_t observation = 1U; observation <= observation_count; ++observation) {
            if (visitor(
                path, observation, paths[path * width + observation * stride], context
            )) break;
        }
    }
}

std::vector<double> generate_rough_heston_observation_spots(
    const RoughHestonModel& model, const TimeGrid& grid,
    const SimulationConfig& simulation, std::size_t observation_count
) {
    std::vector<double> values(simulation.num_paths * observation_count);
    struct Context { double* data; std::size_t count; } context{values.data(), observation_count};
    const auto store = [](std::size_t path, std::size_t observation, double spot, void* raw) {
        auto& target = *static_cast<Context*>(raw);
        target.data[path * target.count + observation - 1U] = spot;
        return false;
    };
    visit_rough_heston_observation_spots(
        model, grid, simulation, observation_count, store, &context
    );
    return values;
}

}  // namespace ai_factory::simulation
