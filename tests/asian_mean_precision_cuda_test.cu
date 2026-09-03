// Qualify arithmetic and geometric Asian path accumulators on CUDA.
#include "common/check_cuda.cuh"
#include "common/compensated_sum.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using ai_factory::workbench::check_cuda;
using ai_factory::workbench::CompensatedFloatSum;

enum class MeanKind {
    arithmetic,
    geometric
};

enum class SumStrategy {
    fp64,
    fp32,
    compensated_fp32,
    chunked_fp32
};

struct MeanResult {
    double mean_coordinate;
    float published_mean;
};

__host__ __device__ float arithmetic_observation(
    std::size_t case_index,
    std::size_t observation,
    std::size_t row = 0U
) {
    const std::size_t shifted = observation + row % 31U;
    const int centered_31 = static_cast<int>(shifted % 31U) - 15;
    const int centered_7 = static_cast<int>(shifted % 7U) - 3;
    switch (case_index) {
        case 0U:
            return 100.0f + 0.125f * static_cast<float>(centered_31);
        case 1U:
            return shifted % 127U == 0U
                ? 2048.0f
                : 0.03125f * static_cast<float>(1U + shifted % 31U);
        case 2U:
            return 1000.0f
                + 0.0001220703125f * static_cast<float>(centered_7);
        default:
            switch (shifted % 4U) {
                case 0U: return 0.0009765625f;
                case 1U: return 4096.0f;
                case 2U: return 0.015625f;
                default: return 512.0f;
            }
    }
}

__host__ __device__ float log_spot_observation(
    std::size_t case_index,
    std::size_t observation,
    std::size_t row = 0U
) {
    const std::size_t shifted = observation + row % 29U;
    const int centered_31 = static_cast<int>(shifted % 31U) - 15;
    const int centered_7 = static_cast<int>(shifted % 7U) - 3;
    switch (case_index) {
        case 0U:
            return 4.6f + 0.001f * static_cast<float>(centered_31);
        case 1U:
            return -6.0f + 0.5f * static_cast<float>(shifted % 29U);
        case 2U:
            return 7.5f
                + 0.000000476837158203125f
                    * static_cast<float>(centered_7);
        default:
            switch (shifted % 4U) {
                case 0U: return -40.0f;
                case 1U: return 40.0f;
                case 2U: return -0.001f;
                default: return 0.001f;
            }
    }
}

template<MeanKind Kind>
__host__ __device__ float observation_value(
    std::size_t case_index,
    std::size_t observation,
    std::size_t row = 0U
) {
    if constexpr (Kind == MeanKind::arithmetic) {
        return arithmetic_observation(case_index, observation, row);
    }
    return log_spot_observation(case_index, observation, row);
}

template<MeanKind Kind, SumStrategy Strategy>
__device__ MeanResult compute_mean(
    std::size_t case_index,
    std::size_t observation_count,
    std::size_t row
) {
    double fp64_sum = 0.0;
    float fp32_sum = 0.0f;
    CompensatedFloatSum compensated;
    float chunk = 0.0f;
    double chunk_total = 0.0;
    unsigned int chunk_count = 0U;
    for (std::size_t observation = 0U;
         observation < observation_count;
         ++observation) {
        const float value = observation_value<Kind>(
            case_index, observation, row
        );
        if constexpr (Strategy == SumStrategy::fp64) {
            fp64_sum += static_cast<double>(value);
        } else if constexpr (Strategy == SumStrategy::fp32) {
            fp32_sum += value;
        } else if constexpr (Strategy == SumStrategy::compensated_fp32) {
            compensated.add(value);
        } else {
            chunk += value;
            ++chunk_count;
            if (chunk_count == 32U) {
                chunk_total += static_cast<double>(chunk);
                chunk = 0.0f;
                chunk_count = 0U;
            }
        }
    }
    if constexpr (Strategy == SumStrategy::chunked_fp32) {
        chunk_total += static_cast<double>(chunk);
    }

    double mean = 0.0;
    if constexpr (Strategy == SumStrategy::fp64) {
        mean = fp64_sum / static_cast<double>(observation_count);
    } else if constexpr (Strategy == SumStrategy::fp32) {
        mean = static_cast<double>(
            fp32_sum / static_cast<float>(observation_count)
        );
    } else if constexpr (Strategy == SumStrategy::compensated_fp32) {
        mean = static_cast<double>(
            compensated.value() / static_cast<float>(observation_count)
        );
    } else {
        mean = chunk_total / static_cast<double>(observation_count);
    }
    const float published = Kind == MeanKind::arithmetic
        ? static_cast<float>(mean)
        : expf(static_cast<float>(mean));
    return {mean, published};
}

template<MeanKind Kind, SumStrategy Strategy>
__global__ void mean_kernel(
    std::size_t case_index,
    std::size_t observation_count,
    std::size_t row_count,
    MeanResult* __restrict__ output
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < row_count) {
        output[row] = compute_mean<Kind, Strategy>(
            case_index, observation_count, row
        );
    }
}

template<MeanKind Kind, SumStrategy Strategy>
void launch_mean(
    std::size_t case_index,
    std::size_t observation_count,
    std::size_t row_count,
    MeanResult* device_output
) {
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = static_cast<unsigned int>(
        (row_count + threads - 1U) / threads
    );
    mean_kernel<Kind, Strategy><<<blocks, threads>>>(
        case_index, observation_count, row_count, device_output
    );
    check_cuda(cudaGetLastError(), "launch Asian mean strategy");
}

template<typename Launch>
float measure_milliseconds(Launch&& launch) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create Asian benchmark start event");
    check_cuda(cudaEventCreate(&stop), "create Asian benchmark stop event");
    for (int iteration = 0; iteration < 2; ++iteration) launch();
    check_cuda(cudaDeviceSynchronize(), "warm up Asian mean benchmark");
    check_cuda(cudaEventRecord(start), "record Asian benchmark start event");
    for (int iteration = 0; iteration < 8; ++iteration) launch();
    check_cuda(cudaEventRecord(stop), "record Asian benchmark stop event");
    check_cuda(cudaEventSynchronize(stop), "wait for Asian mean benchmark");
    float milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start, stop),
        "measure Asian mean benchmark"
    );
    check_cuda(cudaEventDestroy(stop), "destroy Asian benchmark stop event");
    check_cuda(cudaEventDestroy(start), "destroy Asian benchmark start event");
    return milliseconds / 8.0f;
}

template<MeanKind Kind, SumStrategy Strategy>
float benchmark_strategy(
    std::size_t observation_count,
    std::size_t row_count,
    MeanResult* device_output
) {
    return measure_milliseconds([&] {
        launch_mean<Kind, Strategy>(
            1U, observation_count, row_count, device_output
        );
    });
}

template<MeanKind Kind>
MeanResult reference_mean(
    std::size_t case_index,
    std::size_t observation_count
) {
    long double sum = 0.0L;
    for (std::size_t observation = 0U;
         observation < observation_count;
         ++observation) {
        sum += static_cast<long double>(
            observation_value<Kind>(case_index, observation)
        );
    }
    const long double mean =
        sum / static_cast<long double>(observation_count);
    return {
        static_cast<double>(mean),
        Kind == MeanKind::arithmetic
            ? static_cast<float>(mean)
            : static_cast<float>(std::exp(mean))
    };
}

double relative_error(double observed, double expected) {
    return std::fabs(observed - expected)
        / std::max(std::fabs(expected), std::numeric_limits<double>::min());
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<MeanKind Kind>
void qualify_kind(
    const char* kind_name,
    MeanResult* device_output,
    double& maximum_compensated_coordinate_error,
    double& maximum_compensated_published_error,
    double& maximum_simple_coordinate_error
) {
    constexpr std::array<std::size_t, 4U> observation_counts = {
        17U, 253U, 1765U, 4097U
    };
    constexpr std::array<const char*, 4U> strategy_names = {
        "fp64", "fp32", "compensated_fp32", "chunked_fp32"
    };
    for (std::size_t case_index = 0U; case_index < 4U; ++case_index) {
        const std::size_t count = observation_counts[case_index];
        launch_mean<Kind, SumStrategy::fp64>(
            case_index, count, 1U, device_output + 0U
        );
        launch_mean<Kind, SumStrategy::fp32>(
            case_index, count, 1U, device_output + 1U
        );
        launch_mean<Kind, SumStrategy::compensated_fp32>(
            case_index, count, 1U, device_output + 2U
        );
        launch_mean<Kind, SumStrategy::chunked_fp32>(
            case_index, count, 1U, device_output + 3U
        );
        std::array<MeanResult, 4U> observed{};
        check_cuda(
            cudaMemcpy(
                observed.data(),
                device_output,
                observed.size() * sizeof(MeanResult),
                cudaMemcpyDeviceToHost
            ),
            "copy Asian numeric sweep"
        );
        const MeanResult reference = reference_mean<Kind>(case_index, count);
        for (std::size_t strategy = 0U; strategy < observed.size(); ++strategy) {
            std::cout << "ASIAN_NUMERIC kind=" << kind_name
                      << " case=" << case_index
                      << " observations=" << count
                      << " strategy=" << strategy_names[strategy]
                      << " coordinate=" << std::setprecision(12)
                      << observed[strategy].mean_coordinate
                      << " coordinate_relative_error="
                      << relative_error(
                          observed[strategy].mean_coordinate,
                          reference.mean_coordinate
                      )
                      << " published=" << observed[strategy].published_mean
                      << " published_relative_error="
                      << relative_error(
                          observed[strategy].published_mean,
                          reference.published_mean
                      ) << '\n';
        }
        maximum_compensated_coordinate_error = std::max(
            maximum_compensated_coordinate_error,
            relative_error(
                observed[2].mean_coordinate, reference.mean_coordinate
            )
        );
        maximum_compensated_published_error = std::max(
            maximum_compensated_published_error,
            relative_error(
                observed[2].published_mean, reference.published_mean
            )
        );
        maximum_simple_coordinate_error = std::max(
            maximum_simple_coordinate_error,
            relative_error(
                observed[1].mean_coordinate, reference.mean_coordinate
            )
        );
    }
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t count_status = cudaGetDeviceCount(&device_count);
        if (count_status == cudaErrorNoDevice || device_count == 0) return 77;
        check_cuda(count_status, "query CUDA devices");

        MeanResult* device_output = nullptr;
        check_cuda(
            cudaMalloc(&device_output, 4U * sizeof(MeanResult)),
            "allocate Asian numeric output"
        );
        double arithmetic_coordinate_error = 0.0;
        double arithmetic_published_error = 0.0;
        double arithmetic_simple_error = 0.0;
        qualify_kind<MeanKind::arithmetic>(
            "arithmetic",
            device_output,
            arithmetic_coordinate_error,
            arithmetic_published_error,
            arithmetic_simple_error
        );
        double geometric_coordinate_error = 0.0;
        double geometric_published_error = 0.0;
        double geometric_simple_error = 0.0;
        qualify_kind<MeanKind::geometric>(
            "geometric",
            device_output,
            geometric_coordinate_error,
            geometric_published_error,
            geometric_simple_error
        );
        require(
            arithmetic_coordinate_error < 2.0e-7
                && arithmetic_published_error < 2.0e-7,
            "Compensated FP32 arithmetic mean exceeded its error budget."
        );
        require(
            geometric_coordinate_error < 2.0e-7
                && geometric_published_error < 2.0e-6,
            "Compensated FP32 geometric mean exceeded its error budget."
        );
        require(
            arithmetic_simple_error > arithmetic_coordinate_error
                || geometric_simple_error > geometric_coordinate_error,
            "The sweep did not distinguish simple and compensated FP32."
        );
        check_cuda(cudaFree(device_output), "free Asian numeric output");

        constexpr std::size_t benchmark_rows = 8192U;
        constexpr std::size_t benchmark_observations = 1765U;
        check_cuda(
            cudaMalloc(
                &device_output,
                benchmark_rows * sizeof(MeanResult)
            ),
            "allocate Asian benchmark output"
        );
        const float arithmetic_fp64 = benchmark_strategy<
            MeanKind::arithmetic, SumStrategy::fp64
        >(benchmark_observations, benchmark_rows, device_output);
        const float arithmetic_fp32 = benchmark_strategy<
            MeanKind::arithmetic, SumStrategy::fp32
        >(benchmark_observations, benchmark_rows, device_output);
        const float arithmetic_compensated = benchmark_strategy<
            MeanKind::arithmetic, SumStrategy::compensated_fp32
        >(benchmark_observations, benchmark_rows, device_output);
        const float arithmetic_chunked = benchmark_strategy<
            MeanKind::arithmetic, SumStrategy::chunked_fp32
        >(benchmark_observations, benchmark_rows, device_output);
        const float geometric_fp64 = benchmark_strategy<
            MeanKind::geometric, SumStrategy::fp64
        >(benchmark_observations, benchmark_rows, device_output);
        const float geometric_fp32 = benchmark_strategy<
            MeanKind::geometric, SumStrategy::fp32
        >(benchmark_observations, benchmark_rows, device_output);
        const float geometric_compensated = benchmark_strategy<
            MeanKind::geometric, SumStrategy::compensated_fp32
        >(benchmark_observations, benchmark_rows, device_output);
        const float geometric_chunked = benchmark_strategy<
            MeanKind::geometric, SumStrategy::chunked_fp32
        >(benchmark_observations, benchmark_rows, device_output);
        std::cout << "ASIAN_PERFORMANCE rows=" << benchmark_rows
                  << " observations=" << benchmark_observations
                  << " arithmetic_fp64_ms=" << arithmetic_fp64
                  << " arithmetic_fp32_ms=" << arithmetic_fp32
                  << " arithmetic_compensated_fp32_ms="
                  << arithmetic_compensated
                  << " arithmetic_chunked_fp32_ms=" << arithmetic_chunked
                  << " geometric_fp64_ms=" << geometric_fp64
                  << " geometric_fp32_ms=" << geometric_fp32
                  << " geometric_compensated_fp32_ms="
                  << geometric_compensated
                  << " geometric_chunked_fp32_ms=" << geometric_chunked
                  << '\n';
        check_cuda(cudaFree(device_output), "free Asian benchmark output");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
