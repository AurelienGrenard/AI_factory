#include "ai_factory/cpu/black_scholes/american_puts.hpp"

#include "ai_factory/cpu/black_scholes/common.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/payoffs/american_put.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace ai_factory::cpu::black_scholes {
namespace {

constexpr std::size_t kBasisSize = 4U;

simulation::BlackScholesModel to_model(const cuda::BlackScholesRow& row) {
    return {row.spot, row.risk_free_rate, row.dividend_yield, row.volatility};
}

simulation::TimeGrid to_time_grid(const cuda::BlackScholesRow& row, std::size_t num_steps) {
    return {row.maturity, num_steps};
}

simulation::SimulationConfig to_simulation(const cuda::BlackScholesRow& row, std::size_t num_paths) {
    return {row.seed, num_paths, simulation::kPhilox4x32_10BoxMuller};
}

std::array<double, kBasisSize> laguerre_basis(double x) {
    return {
        1.0,
        1.0 - x,
        1.0 - 2.0 * x + 0.5 * x * x,
        1.0 - 3.0 * x + 1.5 * x * x - x * x * x / 6.0,
    };
}

bool solve_linear_4x4(
    std::array<std::array<double, kBasisSize>, kBasisSize> matrix,
    std::array<double, kBasisSize> rhs,
    std::array<double, kBasisSize>& solution
) {
    for (std::size_t pivot = 0; pivot < kBasisSize; ++pivot) {
        std::size_t best = pivot;
        double best_abs = std::abs(matrix[pivot][pivot]);
        for (std::size_t row = pivot + 1U; row < kBasisSize; ++row) {
            const double candidate = std::abs(matrix[row][pivot]);
            if (candidate > best_abs) {
                best = row;
                best_abs = candidate;
            }
        }
        if (best_abs < 1.0e-14) {
            return false;
        }
        if (best != pivot) {
            std::swap(matrix[pivot], matrix[best]);
            std::swap(rhs[pivot], rhs[best]);
        }
        const double diagonal = matrix[pivot][pivot];
        for (std::size_t column = pivot; column < kBasisSize; ++column) {
            matrix[pivot][column] /= diagonal;
        }
        rhs[pivot] /= diagonal;
        for (std::size_t row = 0; row < kBasisSize; ++row) {
            if (row == pivot) {
                continue;
            }
            const double factor = matrix[row][pivot];
            for (std::size_t column = pivot; column < kBasisSize; ++column) {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    solution = rhs;
    return true;
}

double standard_error(double sumsq, double mean, std::size_t num_paths) {
    const double path_count = static_cast<double>(num_paths);
    const double variance =
        (sumsq - path_count * mean * mean) / static_cast<double>(num_paths - 1U);
    return std::sqrt(std::max(variance, 0.0)) / std::sqrt(path_count);
}

void price_from_paths(
    const std::vector<double>& paths,
    std::size_t num_paths,
    std::size_t num_steps,
    double strike,
    double maturity,
    double rate,
    ai_factory::cuda::MonteCarloOutput& output
) {
    if (paths.size() != num_paths * (num_steps + 1U)) {
        throw std::invalid_argument("Unexpected Black-Scholes path buffer size.");
    }
    const double dt = maturity / static_cast<double>(num_steps);
    std::vector<double> cashflows(num_paths);
    std::vector<std::size_t> exercise_steps(num_paths, num_steps);
    for (std::size_t path = 0; path < num_paths; ++path) {
        cashflows[path] = payoffs::american_put_immediate(
            paths[path * (num_steps + 1U) + num_steps],
            strike
        );
    }
    for (std::size_t reverse_step = num_steps - 1U; reverse_step > 0U; --reverse_step) {
        std::array<std::array<double, kBasisSize>, kBasisSize> normal{};
        std::array<double, kBasisSize> rhs{};
        std::size_t itm_count = 0U;
        for (std::size_t path = 0; path < num_paths; ++path) {
            const double spot = paths[path * (num_steps + 1U) + reverse_step];
            const double immediate = payoffs::american_put_immediate(spot, strike);
            if (immediate <= 0.0) {
                continue;
            }
            ++itm_count;
            const auto basis = laguerre_basis(spot / strike);
            const double target = cashflows[path]
                                  * std::exp(
                                      -rate * dt
                                      * static_cast<double>(exercise_steps[path] - reverse_step)
                                  );
            for (std::size_t i = 0; i < kBasisSize; ++i) {
                rhs[i] += basis[i] * target;
                for (std::size_t j = 0; j < kBasisSize; ++j) {
                    normal[i][j] += basis[i] * basis[j];
                }
            }
        }
        if (itm_count <= kBasisSize) {
            continue;
        }
        std::array<double, kBasisSize> coefficients{};
        if (!solve_linear_4x4(normal, rhs, coefficients)) {
            continue;
        }
        for (std::size_t path = 0; path < num_paths; ++path) {
            const double spot = paths[path * (num_steps + 1U) + reverse_step];
            const double immediate = payoffs::american_put_immediate(spot, strike);
            if (immediate <= 0.0) {
                continue;
            }
            const auto basis = laguerre_basis(spot / strike);
            double continuation = 0.0;
            for (std::size_t i = 0; i < kBasisSize; ++i) {
                continuation += coefficients[i] * basis[i];
            }
            if (immediate > continuation) {
                cashflows[path] = immediate;
                exercise_steps[path] = reverse_step;
            }
        }
    }
    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t path = 0; path < num_paths; ++path) {
        const double discounted =
            cashflows[path] * std::exp(-rate * dt * static_cast<double>(exercise_steps[path]));
        sum += discounted;
        sumsq += discounted * discounted;
    }
    const double mean = sum / static_cast<double>(num_paths);
    output = {mean, standard_error(sumsq, mean, num_paths)};
}

}  // namespace

void price_american_put_batch(
    const cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t index = 0; index < static_cast<std::ptrdiff_t>(row_count); ++index) {
        const auto& row = rows[static_cast<std::size_t>(index)];
        const auto paths = simulation::generate_black_scholes_spot_paths(
            to_model(row), to_time_grid(row, num_steps), to_simulation(row, num_paths)
        );
        price_from_paths(
            paths,
            num_paths,
            num_steps,
            row.strike,
            row.maturity,
            row.risk_free_rate,
            outputs[static_cast<std::size_t>(index)]
        );
    }
}

}  // namespace ai_factory::cpu::black_scholes
