#pragma once

#include "ai_factory/cpu/common/lsm.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace ai_factory::cpu::lsm {

struct ExerciseValue {
    double immediate;
    double basis_state;
};

template <typename ExerciseEvaluator>
std::vector<double> bermudan_cashflows_present_value(
    const std::vector<double>& states,
    const std::vector<double>& discounts,
    std::size_t exercise_count,
    std::size_t num_paths,
    ExerciseEvaluator&& evaluate
) {
    std::vector<double> cashflows(num_paths, 0.0);
    const std::size_t last_exercise = exercise_count - 1U;
    for (std::size_t path = 0; path < num_paths; ++path) {
        const auto value = evaluate(
            last_exercise,
            states[last_exercise * num_paths + path]
        );
        cashflows[path] = discounts[last_exercise * num_paths + path]
                          * value.immediate;
    }

    for (std::size_t exercise = last_exercise; exercise-- > 0U;) {
        Matrix normal{};
        Basis rhs{};
        std::size_t itm_count = 0U;
        for (std::size_t path = 0; path < num_paths; ++path) {
            const std::size_t index = exercise * num_paths + path;
            const auto value = evaluate(exercise, states[index]);
            if (value.immediate <= 0.0) {
                continue;
            }
            const auto basis = laguerre_basis(value.basis_state);
            const double target = cashflows[path] / discounts[index];
            for (std::size_t row = 0; row < kBasisSize; ++row) {
                rhs[row] += basis[row] * target;
                for (std::size_t column = 0; column < kBasisSize; ++column) {
                    normal[row][column] += basis[row] * basis[column];
                }
            }
            ++itm_count;
        }

        Basis coefficients{};
        const bool valid = itm_count > kBasisSize
                           && solve_linear_4x4(normal, rhs, coefficients);
        if (!valid) {
            continue;
        }
        for (std::size_t path = 0; path < num_paths; ++path) {
            const std::size_t index = exercise * num_paths + path;
            const auto value = evaluate(exercise, states[index]);
            if (value.immediate <= 0.0) {
                continue;
            }
            const auto basis = laguerre_basis(value.basis_state);
            double continuation = 0.0;
            for (std::size_t basis_index = 0;
                 basis_index < kBasisSize;
                 ++basis_index) {
                continuation += coefficients[basis_index] * basis[basis_index];
            }
            if (value.immediate > continuation) {
                cashflows[path] = discounts[index] * value.immediate;
            }
        }
    }
    return cashflows;
}

inline std::vector<double> bermudan_cashflows_present_value(
    const std::vector<double>& immediate,
    const std::vector<double>& basis_states,
    const std::vector<double>& discounts,
    std::size_t exercise_count,
    std::size_t num_paths
) {
    const auto evaluate = [&](std::size_t exercise, double encoded_path) {
        const auto path = static_cast<std::size_t>(encoded_path);
        const auto index = exercise * num_paths + path;
        return ExerciseValue{immediate[index], basis_states[index]};
    };
    std::vector<double> path_ids(exercise_count * num_paths);
    for (std::size_t exercise = 0; exercise < exercise_count; ++exercise) {
        for (std::size_t path = 0; path < num_paths; ++path) {
            path_ids[exercise * num_paths + path] = static_cast<double>(path);
        }
    }
    return bermudan_cashflows_present_value(
        path_ids, discounts, exercise_count, num_paths, evaluate
    );
}

}  // namespace ai_factory::cpu::lsm
