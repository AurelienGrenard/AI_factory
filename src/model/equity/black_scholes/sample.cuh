// Persistent CUDA launchers for generated Black-Scholes training samples.
#pragma once

#include "common/sample.cuh"
#include "model/equity/black_scholes/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::black_scholes {

struct BlackScholesSampleBounds {
    sample::UniformBounds spot;
    sample::UniformBounds risk_free_rate;
    sample::UniformBounds dividend_yield;
    sample::UniformBounds volatility;
};

struct BlackScholesSampleValues {
    float spot;
};

using BlackScholesTerminalSampleRow = sample::TerminalSampleRow<
    ModelParameters,
    BlackScholesSampleValues
>;

template <std::uint32_t ObservationCount>
using BlackScholesCalendarSampleRow = sample::CalendarSampleRow<
    ModelParameters,
    BlackScholesSampleValues,
    ObservationCount
>;

static_assert(std::is_trivially_copyable_v<BlackScholesSampleBounds>);
static_assert(std::is_trivially_copyable_v<BlackScholesTerminalSampleRow>);
static_assert(
    sizeof(BlackScholesTerminalSampleRow) == 6U * sizeof(float),
    "A terminal Black-Scholes row must be four parameters, T, and S_T."
);

void launch_black_scholes_terminal_samples_cuda(
    BlackScholesSampleBounds parameter_bounds,
    sample::UniformBounds maturity_bounds,
    time_grid::TimeGrid grid,
    std::size_t total_sample_count,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    BlackScholesTerminalSampleRow* device_rows
);

void launch_black_scholes_random_calendar_samples_cuda(
    BlackScholesSampleBounds parameter_bounds,
    sample::RandomCalendarRules calendar_rules,
    time_grid::TimeGrid grid,
    std::uint32_t observation_count,
    std::size_t total_sample_count,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    void* device_rows
);

void launch_black_scholes_fixed_calendar_samples_cuda(
    BlackScholesSampleBounds parameter_bounds,
    const float* observation_times,
    time_grid::TimeGrid grid,
    std::uint32_t observation_count,
    std::size_t total_sample_count,
    std::size_t paths_per_parameter,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    void* device_rows
);

}  // namespace ai_factory::workbench::black_scholes
