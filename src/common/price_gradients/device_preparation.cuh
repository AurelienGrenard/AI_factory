// Generic finite-difference task preparation over a scenario adapter.
#pragma once

#include "common/price_gradients/reconstruction.cuh"
#include "common/price_gradients/time_configuration.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::price_gradients::device_preparation {

namespace pg = ::ai_factory::workbench::price_gradients;

struct Error {
    int code;
    std::uint64_t row;
    std::uint32_t sensitivity;
};

enum ErrorCode : int {
    valid = 0,
    invalid_central = 1,
    invalid_bump = 2,
    unrepresentable_bump = 3,
    no_admissible_stencil = 4,
    invalid_maturity_bump = 5,
    unsupported_order = 6,
};

__host__ __device__ inline bool finite(float value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

__host__ __device__ inline bool finite(double value) {
#if defined(__CUDA_ARCH__)
    return ::isfinite(value);
#else
    return std::isfinite(value);
#endif
}

#if defined(__CUDACC__)
__device__ __forceinline__ void record_error(
    Error* error,
    int code,
    std::size_t row,
    std::size_t sensitivity
) {
    if (error == nullptr) return;
    if (atomicCAS(&error->code, valid, code) == valid) {
        error->row = static_cast<std::uint64_t>(row);
        error->sensitivity = static_cast<std::uint32_t>(sensitivity);
    }
}
#endif

struct RegularEndpoint {
    float central;

    __host__ __device__ float operator()(int multiple, float h) const {
        return static_cast<float>(
            static_cast<double>(central)
            + static_cast<double>(multiple) * static_cast<double>(h)
        );
    }
};

struct MaturityEndpoint {
    std::uint32_t central_steps;
    std::int64_t bump_steps;
    pg::TimeConfiguration time;

    __host__ __device__ float operator()(int multiple, float) const {
        const auto endpoint_steps = static_cast<std::int64_t>(central_steps)
            + static_cast<std::int64_t>(multiple) * bump_steps;
        return static_cast<float>(endpoint_steps) * time.dt;
    }
};

template<std::size_t NodeCapacity>
__host__ __device__ inline bool finite_difference_weights(
    const float* nodes,
    std::size_t node_count,
    unsigned int derivative_order,
    float* weights
) {
    double matrix[4U][5U]{};
    const double origin = static_cast<double>(nodes[0U]);
    for (std::size_t power = 0U; power < node_count; ++power) {
        for (std::size_t node = 0U; node < node_count; ++node) {
            const double x = static_cast<double>(nodes[node]) - origin;
            double value = 1.0;
            for (std::size_t p = 0U; p < power; ++p) value *= x;
            matrix[power][node] = value;
        }
        matrix[power][node_count] = power == derivative_order
            ? (derivative_order == 2U ? 2.0 : 1.0)
            : 0.0;
    }
    for (std::size_t column = 0U; column < node_count; ++column) {
        std::size_t pivot = column;
        double pivot_size = ::fabs(matrix[pivot][column]);
        for (std::size_t row = column + 1U; row < node_count; ++row) {
            const double candidate = ::fabs(matrix[row][column]);
            if (candidate > pivot_size) {
                pivot = row;
                pivot_size = candidate;
            }
        }
        if (!(pivot_size > 0.0) || !finite(pivot_size)) return false;
        if (pivot != column) {
            for (std::size_t entry = column; entry <= node_count; ++entry) {
                const double temporary = matrix[column][entry];
                matrix[column][entry] = matrix[pivot][entry];
                matrix[pivot][entry] = temporary;
            }
        }
        const double divisor = matrix[column][column];
        for (std::size_t entry = column; entry <= node_count; ++entry) {
            matrix[column][entry] /= divisor;
        }
        for (std::size_t row = 0U; row < node_count; ++row) {
            if (row == column) continue;
            const double factor = matrix[row][column];
            for (std::size_t entry = column; entry <= node_count; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
        }
    }
    for (std::size_t node = 0U; node < node_count; ++node) {
        weights[node] = static_cast<float>(matrix[node][node_count]);
        if (!finite(weights[node])) return false;
    }
    for (std::size_t node = node_count; node < NodeCapacity; ++node) {
        weights[node] = 0.0f;
    }
    return true;
}

template<pg::SensitivityOrders Orders, typename Preparation, typename Endpoint>
__host__ __device__ inline bool prepare_stencil(
    float central,
    pg::BumpConfiguration configuration,
    const typename Preparation::Scenario& central_row,
    typename Preparation::Parameter parameter,
    pg::TimeConfiguration time,
    Endpoint endpoint,
    pg::SensitivityStencil<pg::SensitivityTraits<Orders>::node_capacity>& result,
    int& error
) {
    constexpr std::size_t capacity = pg::SensitivityTraits<Orders>::node_capacity;
    typename Preparation::Scenario candidate{};
    const auto admissible = [&](float value) {
        return finite(value)
            && Preparation::change_scenario(
                central_row, parameter, value, time, candidate
            );
    };
    if (!finite(central) || !admissible(central)) {
        error = invalid_central;
        return false;
    }
    if (!finite(configuration.displacement)
        || !(configuration.displacement > 0.0)
        || (configuration.scale != pg::BumpScale::absolute
            && configuration.scale != pg::BumpScale::relative)
        || (configuration.boundary != pg::BoundaryRule::central_only
            && configuration.boundary
                != pg::BoundaryRule::central_then_one_sided_order2)) {
        error = invalid_bump;
        return false;
    }
    const float h = configuration.scale == pg::BumpScale::relative
        ? static_cast<float>(configuration.displacement) * ::fabsf(central)
        : static_cast<float>(configuration.displacement);
    if (!finite(h) || !(h > 0.0f)) {
        error = invalid_bump;
        return false;
    }
    const float lower = endpoint(-1, h);
    const float upper = endpoint(1, h);
    if (!(lower < central) || !(central < upper)) {
        error = unrepresentable_bump;
        return false;
    }
    const float width = upper - lower;
    if (admissible(lower) && admissible(upper) && finite(width)) {
        result.kind = pg::StencilKind::centered;
        result.parameter_values[0U] = central;
        result.parameter_values[1U] = lower;
        result.parameter_values[2U] = upper;
        result.displacement = h;
        result.represented_width = width;
        result.first_endpoint_weights[0U] = -1.0f / width;
        result.first_endpoint_weights[1U] = 1.0f / width;
        if constexpr (capacity == 4U) {
            result.node_count = 3U;
            if (!finite_difference_weights<capacity>(
                    result.parameter_values, 3U, 2U, result.second_weights
                )) {
                error = unrepresentable_bump;
                return false;
            }
        }
        return true;
    }
    if (configuration.boundary == pg::BoundaryRule::central_only) {
        error = no_admissible_stencil;
        return false;
    }
    for (int direction = 1; direction >= -1; direction -= 2) {
        const float first = endpoint(direction, h);
        const float second = endpoint(2 * direction, h);
        const float third = endpoint(3 * direction, h);
        if (!admissible(first) || !admissible(second) || first == second) {
            continue;
        }
        if constexpr (capacity == 4U) {
            if (!admissible(third) || second == third) continue;
        }
        const double a = static_cast<double>(first)
            - static_cast<double>(central);
        const double b = static_cast<double>(second)
            - static_cast<double>(central);
        const float first_weight = static_cast<float>(b / (a * (b - a)));
        const float second_weight = static_cast<float>(-a / (b * (b - a)));
        if (!finite(first_weight) || !finite(second_weight)) continue;
        result.kind = direction > 0
            ? pg::StencilKind::forward
            : pg::StencilKind::backward;
        result.parameter_values[0U] = central;
        result.parameter_values[1U] = first;
        result.parameter_values[2U] = second;
        result.displacement = h;
        result.represented_width = second - first;
        result.first_endpoint_weights[0U] = first_weight;
        result.first_endpoint_weights[1U] = second_weight;
        if constexpr (capacity == 4U) {
            result.parameter_values[3U] = third;
            result.node_count = 4U;
            if (!finite_difference_weights<capacity>(
                    result.parameter_values, 4U, 2U, result.second_weights
                )) {
                continue;
            }
        }
        return true;
    }
    error = no_admissible_stencil;
    return false;
}

template<pg::SensitivityOrders Orders, typename Preparation>
__host__ __device__ inline bool build_stencil(
    const typename Preparation::Scenario& central,
    pg::SensitivitySpec<typename Preparation::Parameter> sensitivity,
    pg::TimeConfiguration time,
    pg::SensitivityStencil<pg::SensitivityTraits<Orders>::node_capacity>& stencil,
    int& error
) {
    if constexpr (Preparation::kSupportsMaturitySensitivity) {
        if (Preparation::is_maturity(sensitivity.parameter)) {
            if constexpr (Orders != pg::SensitivityOrders::first
                && !Preparation::kSupportsMaturityDiagonal) {
                error = unsupported_order;
                return false;
            }
            const float h = sensitivity.bump.scale == pg::BumpScale::relative
                ? static_cast<float>(sensitivity.bump.displacement)
                    * central.maturity_years
                : static_cast<float>(sensitivity.bump.displacement);
            const double step_width = static_cast<double>(h)
                / static_cast<double>(time.dt);
            if (!finite(step_width) || step_width < 1.0
                || step_width > static_cast<double>(0xffffffffU)) {
                error = invalid_maturity_bump;
                return false;
            }
            const auto bump_steps = ::llround(step_width);
            if (static_cast<float>(bump_steps) * time.dt != h) {
                error = invalid_maturity_bump;
                return false;
            }
            return prepare_stencil<Orders, Preparation>(
                central.maturity_years,
                sensitivity.bump,
                central,
                sensitivity.parameter,
                time,
                MaturityEndpoint{
                    central.step_count,
                    static_cast<std::int64_t>(bump_steps),
                    time,
                },
                stencil,
                error
            );
        }
    }
    const float value = Preparation::read_parameter(
        sensitivity.parameter, central
    );
    return prepare_stencil<Orders, Preparation>(
        value,
        sensitivity.bump,
        central,
        sensitivity.parameter,
        time,
        RegularEndpoint{value},
        stencil,
        error
    );
}

template<std::size_t NodeCapacity>
__host__ __device__ inline void brownian_endpoint_weights(
    const float (&input_times)[NodeCapacity],
    std::size_t node_count,
    float (&weights)[NodeCapacity][NodeCapacity]
) {
    double coefficients[NodeCapacity][NodeCapacity]{};
    double times[NodeCapacity]{};
    for (std::size_t node = 0U; node < node_count; ++node) {
        times[node] = static_cast<double>(input_times[node]);
    }
    coefficients[0U][0U] = ::sqrt(times[0U]);
    for (std::size_t index = 1U; index < node_count; ++index) {
        if (times[index] == times[0U]) {
            for (std::size_t normal = 0U; normal < node_count; ++normal) {
                coefficients[index][normal] = coefficients[0U][normal];
            }
            continue;
        }
        bool reused = false;
        for (std::size_t known = 1U; known < index; ++known) {
            if (times[index] != times[known]) continue;
            for (std::size_t normal = 0U; normal < node_count; ++normal) {
                coefficients[index][normal] = coefficients[known][normal];
            }
            reused = true;
            break;
        }
        if (reused) continue;
        int left = -1;
        int right = -1;
        for (std::size_t known = 0U; known < index; ++known) {
            if (times[known] < times[index]
                && (left < 0 || times[known] > times[left])) {
                left = static_cast<int>(known);
            }
            if (times[known] > times[index]
                && (right < 0 || times[known] < times[right])) {
                right = static_cast<int>(known);
            }
        }
        const double left_time = left < 0 ? 0.0 : times[left];
        if (right < 0) {
            if (left >= 0) {
                for (std::size_t normal = 0U;
                     normal < node_count;
                     ++normal) {
                    coefficients[index][normal] = coefficients[left][normal];
                }
            }
            coefficients[index][index] = ::sqrt(times[index] - left_time);
        } else {
            const double interval = times[right] - left_time;
            const double fraction = (times[index] - left_time) / interval;
            for (std::size_t normal = 0U;
                 normal < node_count;
                 ++normal) {
                coefficients[index][normal] = (
                    left < 0
                        ? 0.0
                        : (1.0 - fraction) * coefficients[left][normal]
                ) + fraction * coefficients[right][normal];
            }
            coefficients[index][index] = ::sqrt(
                (times[index] - left_time)
                * (times[right] - times[index]) / interval
            );
        }
    }
    for (std::size_t node = 0U; node < node_count; ++node) {
        const double scale = ::sqrt(times[node]);
        for (std::size_t normal = 0U; normal < node_count; ++normal) {
            weights[node][normal] = static_cast<float>(
                coefficients[node][normal] / scale
            );
        }
    }
}

__host__ __device__ inline void brownian_endpoint_weights(
    float central_time,
    float first_time,
    float second_time,
    float* first_weights,
    float* second_weights
) {
    const float times[3U]{central_time, first_time, second_time};
    float weights[3U][3U]{};
    brownian_endpoint_weights(times, 3U, weights);
    for (std::size_t normal = 0U; normal < 3U; ++normal) {
        first_weights[normal] = weights[1U][normal];
        second_weights[normal] = weights[2U][normal];
    }
}

template<pg::SensitivityOrders Orders, typename Preparation>
__host__ __device__ inline bool build_sensitivity_task(
    const typename Preparation::Scenario& central,
    pg::SensitivitySpec<typename Preparation::Parameter> sensitivity,
    pg::TimeConfiguration time,
    pg::SensitivityTask<
        typename Preparation::Scenario,
        pg::SensitivityTraits<Orders>::node_capacity
    >& task,
    int& error
) {
    task.nodes[0U] = central;
    if (!build_stencil<Orders, Preparation>(
            central, sensitivity, time, task.stencil, error
        )) {
        return false;
    }
    const auto count = pg::active_node_count(task.stencil);
    for (std::size_t node = 1U; node < count; ++node) {
        if (!Preparation::change_scenario(
                central,
                sensitivity.parameter,
                task.stencil.parameter_values[node],
                time,
                task.nodes[node]
            )) {
            error = no_admissible_stencil;
            return false;
        }
    }
    Preparation::template finalize_task<Orders>(
        task, sensitivity.parameter
    );
    return true;
}

static_assert(std::is_trivially_copyable_v<Error>);

}  // namespace ai_factory::workbench::price_gradients::device_preparation
