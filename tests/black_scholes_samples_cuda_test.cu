// Verify generated sample rows, packaged parameters, calendars, and time grids.
#include "common/check_cuda.cuh"
#include "common/time_grid.cuh"
#include "model/equity/black_scholes/sample.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using namespace ai_factory::workbench;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ai_factory::workbench::model::equity::black_scholes::BlackScholesSampleBounds parameter_bounds() {
    return {
        {1.0f, 1.0f},
        {0.001f, 0.08f},
        {0.0f, 0.06f},
        {0.08f, 0.45f},
    };
}

template <typename Row, typename Launch>
std::vector<Row> generate_rows(std::size_t row_count, Launch launch) {
    Row* device_rows = nullptr;
    check_cuda(
        cudaMalloc(&device_rows, row_count * sizeof(Row)),
        "sample test row allocation"
    );
    launch(device_rows);
    check_cuda(cudaDeviceSynchronize(), "sample test synchronize");
    std::vector<Row> rows(row_count);
    check_cuda(
        cudaMemcpy(
            rows.data(),
            device_rows,
            row_count * sizeof(Row),
            cudaMemcpyDeviceToHost
        ),
        "sample test row copy"
    );
    check_cuda(cudaFree(device_rows), "sample test row free");
    return rows;
}

void check_terminal_grid(
    const std::vector<ai_factory::workbench::model::equity::black_scholes::BlackScholesTerminalSampleRow>& rows,
    time_grid::TimeGrid grid,
    std::uint32_t minimum_index,
    std::uint32_t maximum_index
) {
    for (const auto& row : rows) {
        const float scaled =
            row.maturity * static_cast<float>(grid.steps_per_year);
        const float rounded = std::floor(scaled + 0.5f);
        require(
            std::fabs(scaled - rounded) <= 1.0e-4f,
            "sample maturity is not on its configured grid"
        );
        require(
            rounded >= static_cast<float>(minimum_index)
                && rounded <= static_cast<float>(maximum_index),
            "sample maturity lies outside its configured bounds"
        );
        require(
            std::isfinite(row.values.spot) && row.values.spot > 0.0f,
            "sample terminal spot is not finite and positive"
        );
    }
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "sample test cudaGetDeviceCount");

    constexpr std::size_t row_count = 1000U;
    constexpr std::uint64_t seed = 881000001ULL;
    const auto bounds = parameter_bounds();

    const time_grid::TimeGrid daily_grid(360U);
    const sample::UniformBounds daily_maturity{
        90.0f / 360.0f,
        720.0f / 360.0f,
    };
    const auto iid = generate_rows<
        ai_factory::workbench::model::equity::black_scholes::BlackScholesTerminalSampleRow
    >(row_count, [&](auto* device_rows) {
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_terminal_samples_cuda(
            bounds,
            daily_maturity,
            daily_grid,
            row_count,
            1U,
            0U,
            row_count,
            128U,
            4U,
            seed,
            device_rows
        );
    });
    check_terminal_grid(iid, daily_grid, 90U, 720U);

    const auto replay = generate_rows<
        ai_factory::workbench::model::equity::black_scholes::BlackScholesTerminalSampleRow
    >(row_count, [&](auto* device_rows) {
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_terminal_samples_cuda(
            bounds,
            daily_maturity,
            daily_grid,
            row_count,
            1U,
            0U,
            row_count,
            128U,
            4U,
            seed,
            device_rows
        );
    });
    require(
        std::memcmp(
            iid.data(), replay.data(), row_count * sizeof(iid.front())
        ) == 0,
        "sample replay changed for the same seed"
    );

    const auto packaged = generate_rows<
        ai_factory::workbench::model::equity::black_scholes::BlackScholesTerminalSampleRow
    >(row_count, [&](auto* device_rows) {
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_terminal_samples_cuda(
            bounds,
            daily_maturity,
            daily_grid,
            row_count,
            250U,
            0U,
            row_count,
            256U,
            4U,
            seed + 1U,
            device_rows
        );
    });
    check_terminal_grid(packaged, daily_grid, 90U, 720U);
    for (std::size_t package = 0U; package < 4U; ++package) {
        const auto& expected = packaged[package * 250U].parameters;
        for (std::size_t path = 1U; path < 250U; ++path) {
            require(
                std::memcmp(
                    &expected,
                    &packaged[package * 250U + path].parameters,
                    sizeof(expected)
                ) == 0,
                "packaged paths do not share one parameter draw"
            );
        }
    }

    const time_grid::TimeGrid trading_grid(252U);
    const sample::UniformBounds trading_maturity{
        63.0f / 252.0f,
        504.0f / 252.0f,
    };
    const auto trading = generate_rows<
        ai_factory::workbench::model::equity::black_scholes::BlackScholesTerminalSampleRow
    >(row_count, [&](auto* device_rows) {
        ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_terminal_samples_cuda(
            bounds,
            trading_maturity,
            trading_grid,
            row_count,
            1U,
            0U,
            row_count,
            128U,
            4U,
            seed + 2U,
            device_rows
        );
    });
    check_terminal_grid(trading, trading_grid, 63U, 504U);

    using CalendarRow =
        ai_factory::workbench::model::equity::black_scholes::BlackScholesCalendarSampleRow<3U>;
    static_assert(sizeof(CalendarRow) == 10U * sizeof(float));
    const sample::RandomCalendarRules calendar_rules{
        daily_maturity,
        30.0f / 360.0f,
    };
    const auto random_calendar = generate_rows<CalendarRow>(
        row_count,
        [&](auto* device_rows) {
            ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_random_calendar_samples_cuda(
                bounds,
                calendar_rules,
                daily_grid,
                3U,
                row_count,
                1U,
                0U,
                row_count,
                128U,
                4U,
                seed + 3U,
                device_rows
            );
        }
    );
    for (const auto& row : random_calendar) {
        std::uint32_t previous = 0U;
        for (std::uint32_t observation = 0U; observation < 3U; ++observation) {
            const std::uint32_t index = time_grid::index(
                row.observation_times[observation],
                daily_grid,
                "generated random-calendar time"
            );
            if (observation != 0U) {
                require(
                    index - previous >= 30U,
                    "random-calendar minimum interval is violated"
                );
            }
            require(
                std::isfinite(row.values[observation].spot)
                    && row.values[observation].spot > 0.0f,
                "random-calendar spot is not finite and positive"
            );
            previous = index;
        }
    }

    const float fixed_times[3] = {
        90.0f / 360.0f,
        180.0f / 360.0f,
        360.0f / 360.0f,
    };
    const auto fixed_calendar = generate_rows<CalendarRow>(
        row_count,
        [&](auto* device_rows) {
            ai_factory::workbench::model::equity::black_scholes::launch_black_scholes_fixed_calendar_samples_cuda(
                bounds,
                fixed_times,
                daily_grid,
                3U,
                row_count,
                1U,
                0U,
                row_count,
                128U,
                4U,
                seed + 4U,
                device_rows
            );
        }
    );
    for (const auto& row : fixed_calendar) {
        for (std::uint32_t observation = 0U; observation < 3U; ++observation) {
            require(
                row.observation_times[observation]
                    == fixed_times[observation],
                "fixed-calendar output time changed"
            );
            require(
                std::isfinite(row.values[observation].spot)
                    && row.values[observation].spot > 0.0f,
                "fixed-calendar spot is not finite and positive"
            );
        }
    }
}
