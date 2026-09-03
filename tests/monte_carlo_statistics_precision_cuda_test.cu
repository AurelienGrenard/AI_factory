// Qualify Monte Carlo moment formation and finalization precision on CUDA.
#include "common/check_cuda.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ai_factory::workbench::check_cuda;
namespace reductions = ai_factory::workbench::reductions;

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kCaseCount = 4U;

enum class MomentStrategy {
    fp64_product,
    fp64_fma,
    fp32_product,
    scaled_fp32_product,
    compensated_fp32_product
};

enum class FinalizationStrategy {
    fp64,
    mixed_fp32_sqrt,
    fp32
};

struct Statistics {
    double price;
    double standard_error;
};

__host__ __device__ float synthetic_payoff(
    std::size_t case_index,
    std::size_t path
) {
    const int centered_31 = static_cast<int>(path % 31U) - 15;
    const int centered_7 = static_cast<int>(path % 7U) - 3;
    switch (case_index) {
        case 0U:
            return path % 5U == 0U
                ? 0.0f
                : 12.5f + 0.75f * static_cast<float>(centered_31);
        case 1U:
            return path % 257U == 0U
                ? 250000.0f + 32.0f * static_cast<float>(centered_31)
                : 0.125f * static_cast<float>(path % 17U);
        case 2U:
            return 100.0f
                + 0.0009765625f * static_cast<float>(centered_7);
        default:
            return 2048.0f + 0.25f * static_cast<float>(centered_7);
    }
}

__host__ __device__ double payoff_scale(std::size_t case_index) {
    switch (case_index) {
        case 0U: return 32.0;
        case 1U: return 262144.0;
        case 2U: return 128.0;
        default: return 2048.0;
    }
}

template<MomentStrategy Strategy>
__global__ void form_moments_kernel(
    std::size_t case_index,
    std::size_t sample_count,
    reductions::MomentSums* __restrict__ output
) {
    double sum = 0.0;
    double sumsq = 0.0;
    float compensated_sumsq = 0.0f;
    float compensation = 0.0f;
    const double scale = payoff_scale(case_index);
    const float inverse_scale = static_cast<float>(1.0 / scale);

    for (std::size_t path = threadIdx.x;
         path < sample_count;
         path += blockDim.x) {
        const float payoff = synthetic_payoff(case_index, path);
        const double value = static_cast<double>(payoff);
        sum += value;
        if constexpr (Strategy == MomentStrategy::fp64_product) {
            sumsq += value * value;
        } else if constexpr (Strategy == MomentStrategy::fp64_fma) {
            sumsq = fma(value, value, sumsq);
        } else if constexpr (Strategy == MomentStrategy::fp32_product) {
            sumsq += static_cast<double>(payoff * payoff);
        } else if constexpr (
            Strategy == MomentStrategy::scaled_fp32_product
        ) {
            const float scaled = payoff * inverse_scale;
            sumsq += static_cast<double>(scaled * scaled);
        } else {
            const float term = payoff * payoff;
            const float corrected = term - compensation;
            const float updated = compensated_sumsq + corrected;
            compensation = (updated - compensated_sumsq) - corrected;
            compensated_sumsq = updated;
        }
    }
    if constexpr (Strategy == MomentStrategy::compensated_fp32_product) {
        sumsq = static_cast<double>(compensated_sumsq);
    }

    const reductions::MomentSums total = reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        const double restored_sumsq =
            Strategy == MomentStrategy::scaled_fp32_product
            ? total.sumsq * scale * scale
            : total.sumsq;
        output[blockIdx.x] = {total.sum, restored_sumsq};
    }
}

__device__ void compute_mixed_statistics(
    const reductions::MomentSums& total,
    std::size_t sample_count,
    double& price,
    double& standard_error
) {
    const double count = static_cast<double>(sample_count);
    price = total.sum / count;
    const double centered_sum = total.sumsq - count * price * price;
    const double sample_variance = fmax(centered_sum, 0.0) / (count - 1.0);
    standard_error = static_cast<double>(
        sqrtf(static_cast<float>(sample_variance / count))
    );
}

__device__ void compute_fp32_statistics(
    const reductions::MomentSums& total,
    std::size_t sample_count,
    double& price,
    double& standard_error
) {
    const float count = static_cast<float>(sample_count);
    const float sum = static_cast<float>(total.sum);
    const float sumsq = static_cast<float>(total.sumsq);
    const float mean = sum / count;
    const float centered_sum = sumsq - count * mean * mean;
    const float sample_variance = fmaxf(centered_sum, 0.0f) / (count - 1.0f);
    price = static_cast<double>(mean);
    standard_error = static_cast<double>(sqrtf(sample_variance / count));
}

template<FinalizationStrategy Strategy>
__global__ void finalize_statistics_kernel(
    const reductions::MomentSums* __restrict__ moments,
    std::size_t moment_count,
    std::size_t sample_count,
    Statistics* __restrict__ output
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= moment_count) return;
    double price = 0.0;
    double standard_error = 0.0;
    if constexpr (Strategy == FinalizationStrategy::fp64) {
        reductions::compute_statistics(
            moments[index], sample_count, price, standard_error
        );
    } else if constexpr (
        Strategy == FinalizationStrategy::mixed_fp32_sqrt
    ) {
        compute_mixed_statistics(
            moments[index], sample_count, price, standard_error
        );
    } else {
        compute_fp32_statistics(
            moments[index], sample_count, price, standard_error
        );
    }
    output[index] = {price, standard_error};
}

template<typename Launch>
float measure_milliseconds(Launch&& launch, int warmups, int repetitions) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create benchmark start event");
    check_cuda(cudaEventCreate(&stop), "create benchmark stop event");
    for (int iteration = 0; iteration < warmups; ++iteration) launch();
    check_cuda(cudaDeviceSynchronize(), "warm up statistics benchmark");
    check_cuda(cudaEventRecord(start), "record benchmark start event");
    for (int iteration = 0; iteration < repetitions; ++iteration) launch();
    check_cuda(cudaEventRecord(stop), "record benchmark stop event");
    check_cuda(cudaEventSynchronize(stop), "wait for statistics benchmark");
    float milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start, stop),
        "measure statistics benchmark"
    );
    check_cuda(cudaEventDestroy(stop), "destroy benchmark stop event");
    check_cuda(cudaEventDestroy(start), "destroy benchmark start event");
    return milliseconds / static_cast<float>(repetitions);
}

template<MomentStrategy Strategy>
float benchmark_moment_strategy(
    std::size_t case_index,
    std::size_t sample_count,
    std::size_t row_count,
    reductions::MomentSums* device_output
) {
    const std::size_t shared_bytes =
        2U * (kThreads / 32U) * sizeof(double);
    return measure_milliseconds(
        [&] {
            form_moments_kernel<Strategy><<<
                static_cast<unsigned int>(row_count),
                kThreads,
                shared_bytes
            >>>(case_index, sample_count, device_output);
            check_cuda(cudaGetLastError(), "launch moment strategy");
        },
        2,
        8
    );
}

template<FinalizationStrategy Strategy>
float benchmark_finalization_strategy(
    const reductions::MomentSums* device_moments,
    std::size_t moment_count,
    std::size_t sample_count,
    Statistics* device_output
) {
    const unsigned int blocks = static_cast<unsigned int>(
        (moment_count + kThreads - 1U) / kThreads
    );
    return measure_milliseconds(
        [&] {
            finalize_statistics_kernel<Strategy><<<blocks, kThreads>>>(
                device_moments, moment_count, sample_count, device_output
            );
            check_cuda(cudaGetLastError(), "launch finalization strategy");
        },
        5,
        50
    );
}

Statistics reference_statistics(
    std::size_t case_index,
    std::size_t sample_count
) {
    long double sum = 0.0L;
    long double sumsq = 0.0L;
    for (std::size_t path = 0U; path < sample_count; ++path) {
        const long double value = static_cast<long double>(
            synthetic_payoff(case_index, path)
        );
        sum += value;
        sumsq += value * value;
    }
    const long double count = static_cast<long double>(sample_count);
    const long double price = sum / count;
    const long double variance =
        (sumsq - count * price * price) / (count - 1.0L);
    return {
        static_cast<double>(price),
        static_cast<double>(std::sqrt(variance / count))
    };
}

double relative_error(double observed, double expected) {
    return std::fabs(observed - expected)
        / std::max(std::fabs(expected), std::numeric_limits<double>::min());
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t count_status = cudaGetDeviceCount(&device_count);
        if (count_status == cudaErrorNoDevice || device_count == 0) return 77;
        check_cuda(count_status, "query CUDA devices");

        constexpr std::size_t numeric_sample_count = 1U << 20U;
        constexpr std::size_t strategy_count = 5U;
        reductions::MomentSums* device_moments = nullptr;
        Statistics* device_statistics = nullptr;
        check_cuda(
            cudaMalloc(
                &device_moments,
                strategy_count * sizeof(reductions::MomentSums)
            ),
            "allocate numeric moments"
        );
        check_cuda(
            cudaMalloc(
                &device_statistics,
                strategy_count * sizeof(Statistics)
            ),
            "allocate numeric statistics"
        );

        const std::array<const char*, strategy_count> strategy_names = {
            "fp64_product", "fp64_fma", "fp32_product",
            "scaled_fp32_product", "compensated_fp32_product"
        };
        double maximum_fp64_price_error = 0.0;
        double maximum_fp64_standard_error = 0.0;
        double maximum_mixed_standard_error = 0.0;
        double maximum_fp32_standard_error = 0.0;
        for (std::size_t case_index = 0U;
             case_index < kCaseCount;
             ++case_index) {
            const std::size_t shared_bytes =
                2U * (kThreads / 32U) * sizeof(double);
            form_moments_kernel<MomentStrategy::fp64_product><<<
                1U, kThreads, shared_bytes
            >>>(case_index, numeric_sample_count, device_moments + 0U);
            form_moments_kernel<MomentStrategy::fp64_fma><<<
                1U, kThreads, shared_bytes
            >>>(case_index, numeric_sample_count, device_moments + 1U);
            form_moments_kernel<MomentStrategy::fp32_product><<<
                1U, kThreads, shared_bytes
            >>>(case_index, numeric_sample_count, device_moments + 2U);
            form_moments_kernel<MomentStrategy::scaled_fp32_product><<<
                1U, kThreads, shared_bytes
            >>>(case_index, numeric_sample_count, device_moments + 3U);
            form_moments_kernel<MomentStrategy::compensated_fp32_product><<<
                1U, kThreads, shared_bytes
            >>>(case_index, numeric_sample_count, device_moments + 4U);
            check_cuda(cudaGetLastError(), "launch numeric moment sweep");

            finalize_statistics_kernel<FinalizationStrategy::fp64><<<1U, 32U>>>(
                device_moments,
                strategy_count,
                numeric_sample_count,
                device_statistics
            );
            check_cuda(cudaGetLastError(), "launch numeric finalization sweep");
            std::array<Statistics, strategy_count> statistics{};
            check_cuda(
                cudaMemcpy(
                    statistics.data(),
                    device_statistics,
                    strategy_count * sizeof(Statistics),
                    cudaMemcpyDeviceToHost
                ),
                "copy numeric statistics"
            );
            const Statistics reference =
                reference_statistics(case_index, numeric_sample_count);
            const double fp64_price_error =
                relative_error(statistics[0].price, reference.price);
            const double fp64_standard_error = relative_error(
                statistics[0].standard_error, reference.standard_error
            );
            maximum_fp64_price_error =
                std::max(maximum_fp64_price_error, fp64_price_error);
            maximum_fp64_standard_error = std::max(
                maximum_fp64_standard_error, fp64_standard_error
            );
            for (std::size_t strategy = 0U;
                 strategy < strategy_count;
                 ++strategy) {
                std::cout << "MOMENT_NUMERIC case=" << case_index
                          << " strategy=" << strategy_names[strategy]
                          << " price=" << std::setprecision(12)
                          << statistics[strategy].price
                          << " stderr=" << statistics[strategy].standard_error
                          << " stderr_relative_error="
                          << relative_error(
                              statistics[strategy].standard_error,
                              reference.standard_error
                          ) << '\n';
            }

            finalize_statistics_kernel<FinalizationStrategy::mixed_fp32_sqrt>
                <<<1U, 32U>>>(
                    device_moments,
                    1U,
                    numeric_sample_count,
                    device_statistics
                );
            Statistics mixed{};
            check_cuda(
                cudaMemcpy(
                    &mixed,
                    device_statistics,
                    sizeof(Statistics),
                    cudaMemcpyDeviceToHost
                ),
                "copy mixed finalization"
            );
            finalize_statistics_kernel<FinalizationStrategy::fp32>
                <<<1U, 32U>>>(
                    device_moments,
                    1U,
                    numeric_sample_count,
                    device_statistics
                );
            Statistics fp32{};
            check_cuda(
                cudaMemcpy(
                    &fp32,
                    device_statistics,
                    sizeof(Statistics),
                    cudaMemcpyDeviceToHost
                ),
                "copy FP32 finalization"
            );
            const double mixed_standard_error = relative_error(
                mixed.standard_error, reference.standard_error
            );
            const double fp32_standard_error = relative_error(
                fp32.standard_error, reference.standard_error
            );
            maximum_mixed_standard_error = std::max(
                maximum_mixed_standard_error, mixed_standard_error
            );
            maximum_fp32_standard_error = std::max(
                maximum_fp32_standard_error, fp32_standard_error
            );
            std::cout << "FINALIZE_NUMERIC case=" << case_index
                      << " fp64_stderr_relative_error="
                      << fp64_standard_error
                      << " mixed_stderr_relative_error="
                      << mixed_standard_error
                      << " fp32_stderr_relative_error="
                      << fp32_standard_error << '\n';
        }

        require(
            maximum_fp64_price_error < 1.0e-11,
            "FP64 moment formation changed the Monte Carlo price."
        );
        require(
            maximum_fp64_standard_error < 2.0e-4,
            "FP64 statistics exceeded the independent-reference budget."
        );
        require(
            maximum_mixed_standard_error < 2.0e-4,
            "Mixed finalization exceeded the independent-reference budget."
        );
        require(
            maximum_fp32_standard_error > 1.0e-3,
            "The stress sweep did not expose FP32 finalization cancellation."
        );

        check_cuda(cudaFree(device_statistics), "free numeric statistics");
        check_cuda(cudaFree(device_moments), "free numeric moments");

        constexpr std::size_t benchmark_sample_count = 1U << 15U;
        constexpr std::size_t benchmark_row_count = 256U;
        check_cuda(
            cudaMalloc(
                &device_moments,
                benchmark_row_count * sizeof(reductions::MomentSums)
            ),
            "allocate benchmark moments"
        );
        const float fp64_product_ms =
            benchmark_moment_strategy<MomentStrategy::fp64_product>(
                1U,
                benchmark_sample_count,
                benchmark_row_count,
                device_moments
            );
        const float fp64_fma_ms =
            benchmark_moment_strategy<MomentStrategy::fp64_fma>(
                1U,
                benchmark_sample_count,
                benchmark_row_count,
                device_moments
            );
        const float fp32_product_ms =
            benchmark_moment_strategy<MomentStrategy::fp32_product>(
                1U,
                benchmark_sample_count,
                benchmark_row_count,
                device_moments
            );
        const float scaled_fp32_product_ms =
            benchmark_moment_strategy<MomentStrategy::scaled_fp32_product>(
                1U,
                benchmark_sample_count,
                benchmark_row_count,
                device_moments
            );
        const float compensated_fp32_product_ms = benchmark_moment_strategy<
            MomentStrategy::compensated_fp32_product
        >(
            1U,
            benchmark_sample_count,
            benchmark_row_count,
            device_moments
        );
        std::cout << "MOMENT_PERFORMANCE rows=" << benchmark_row_count
                  << " paths=" << benchmark_sample_count
                  << " fp64_product_ms=" << fp64_product_ms
                  << " fp64_fma_ms=" << fp64_fma_ms
                  << " fp32_product_ms=" << fp32_product_ms
                  << " scaled_fp32_product_ms=" << scaled_fp32_product_ms
                  << " compensated_fp32_product_ms="
                  << compensated_fp32_product_ms << '\n';

        constexpr std::size_t finalization_count = 1U << 20U;
        std::vector<reductions::MomentSums> host_moments(finalization_count);
        const Statistics benchmark_reference =
            reference_statistics(2U, numeric_sample_count);
        const long double count =
            static_cast<long double>(numeric_sample_count);
        const long double variance =
            static_cast<long double>(benchmark_reference.standard_error)
            * static_cast<long double>(benchmark_reference.standard_error)
            * count;
        const long double centered_sum = variance * (count - 1.0L);
        const long double sum =
            static_cast<long double>(benchmark_reference.price) * count;
        const long double sumsq = centered_sum + sum * sum / count;
        std::fill(
            host_moments.begin(),
            host_moments.end(),
            reductions::MomentSums{
                static_cast<double>(sum), static_cast<double>(sumsq)
            }
        );
        check_cuda(cudaFree(device_moments), "free moment benchmark output");
        check_cuda(
            cudaMalloc(
                &device_moments,
                finalization_count * sizeof(reductions::MomentSums)
            ),
            "allocate finalization moments"
        );
        check_cuda(
            cudaMemcpy(
                device_moments,
                host_moments.data(),
                finalization_count * sizeof(reductions::MomentSums),
                cudaMemcpyHostToDevice
            ),
            "copy finalization moments"
        );
        check_cuda(
            cudaMalloc(
                &device_statistics,
                finalization_count * sizeof(Statistics)
            ),
            "allocate finalization statistics"
        );
        const float fp64_finalization_ms = benchmark_finalization_strategy<
            FinalizationStrategy::fp64
        >(
            device_moments,
            finalization_count,
            numeric_sample_count,
            device_statistics
        );
        const float mixed_finalization_ms = benchmark_finalization_strategy<
            FinalizationStrategy::mixed_fp32_sqrt
        >(
            device_moments,
            finalization_count,
            numeric_sample_count,
            device_statistics
        );
        const float fp32_finalization_ms = benchmark_finalization_strategy<
            FinalizationStrategy::fp32
        >(
            device_moments,
            finalization_count,
            numeric_sample_count,
            device_statistics
        );

        cudaEvent_t copy_start = nullptr;
        cudaEvent_t copy_stop = nullptr;
        check_cuda(cudaEventCreate(&copy_start), "create host copy start event");
        check_cuda(cudaEventCreate(&copy_stop), "create host copy stop event");
        check_cuda(cudaEventRecord(copy_start), "record host copy start event");
        check_cuda(
            cudaMemcpy(
                host_moments.data(),
                device_moments,
                finalization_count * sizeof(reductions::MomentSums),
                cudaMemcpyDeviceToHost
            ),
            "copy moments for host finalization"
        );
        check_cuda(cudaEventRecord(copy_stop), "record host copy stop event");
        check_cuda(cudaEventSynchronize(copy_stop), "wait for host moment copy");
        float host_copy_ms = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&host_copy_ms, copy_start, copy_stop),
            "measure host moment copy"
        );
        check_cuda(cudaEventDestroy(copy_stop), "destroy host copy stop event");
        check_cuda(cudaEventDestroy(copy_start), "destroy host copy start event");

        std::vector<Statistics> host_statistics(finalization_count);
        const auto host_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0U;
             index < finalization_count;
             ++index) {
            const long double host_count =
                static_cast<long double>(numeric_sample_count);
            const long double host_price =
                static_cast<long double>(host_moments[index].sum) / host_count;
            const long double host_centered =
                static_cast<long double>(host_moments[index].sumsq)
                - host_count * host_price * host_price;
            const long double host_variance =
                std::max(host_centered, 0.0L) / (host_count - 1.0L);
            host_statistics[index] = {
                static_cast<double>(host_price),
                static_cast<double>(std::sqrt(host_variance / host_count))
            };
        }
        const auto host_stop = std::chrono::steady_clock::now();
        const double host_compute_ms =
            std::chrono::duration<double, std::milli>(host_stop - host_start)
                .count();

        std::cout << "FINALIZE_PERFORMANCE rows=" << finalization_count
                  << " fp64_ms=" << fp64_finalization_ms
                  << " mixed_ms=" << mixed_finalization_ms
                  << " fp32_ms=" << fp32_finalization_ms
                  << " host_copy_only_ms=" << host_copy_ms
                  << " host_compute_ms=" << host_compute_ms
                  << " host_total_ms=" << host_copy_ms + host_compute_ms
                  << '\n';

        check_cuda(cudaFree(device_statistics), "free finalization statistics");
        check_cuda(cudaFree(device_moments), "free finalization moments");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
