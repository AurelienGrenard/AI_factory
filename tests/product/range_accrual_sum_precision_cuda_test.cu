// Qualify Black-Scholes range-accrual probability summation on CUDA.
#include "common/check_cuda.cuh"
#include "common/compensated_sum.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"
#include "product/range_accrual/parameters.hpp"

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

using namespace ai_factory::workbench;
namespace black_scholes = model::equity::black_scholes;

enum class SumStrategy {
    fp64,
    fp32,
    compensated_fp32,
    chunked_fp32
};

template<SumStrategy Strategy>
__device__ float range_accrual_price(
    const black_scholes::ModelParameters& model,
    const product::RangeAccrualParameters& product
) {
    constexpr float day_fraction = 1.0f / 252.0f;
    const float interval_years =
        static_cast<float>(product.observation_interval_days) * day_fraction;
    const float maturity_years =
        static_cast<float>(product.maturity_days) * day_fraction;
    const auto evolution = black_scholes::prepare_lognormal_evolution(
        black_scholes::prepare_analytics(model)
    );
    const float log_lower = logf(product.lower_barrier);
    const float log_upper = logf(product.upper_barrier);
    const std::uint32_t observation_count =
        product.maturity_days / product.observation_interval_days;
    double fp64_sum = 0.0;
    float fp32_sum = 0.0f;
    CompensatedFloatSum compensated;
    float chunk = 0.0f;
    double chunk_total = 0.0;
    unsigned int chunk_count = 0U;
    for (std::uint32_t observation = 1U;
         observation <= observation_count;
         ++observation) {
        const float probability =
            black_scholes::lognormal_log_interval_probability(
                evolution,
                log_lower,
                log_upper,
                static_cast<float>(observation) * interval_years
            );
        if constexpr (Strategy == SumStrategy::fp64) {
            fp64_sum += static_cast<double>(probability);
        } else if constexpr (Strategy == SumStrategy::fp32) {
            fp32_sum += probability;
        } else if constexpr (Strategy == SumStrategy::compensated_fp32) {
            compensated.add(probability);
        } else {
            chunk += probability;
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
    float probability_sum = 0.0f;
    if constexpr (Strategy == SumStrategy::fp64) {
        probability_sum = static_cast<float>(fp64_sum);
    } else if constexpr (Strategy == SumStrategy::fp32) {
        probability_sum = fp32_sum;
    } else if constexpr (Strategy == SumStrategy::compensated_fp32) {
        probability_sum = compensated.value();
    } else {
        probability_sum = static_cast<float>(chunk_total);
    }
    const float discount = expf(-model.risk_free_rate * maturity_years);
    return fmaf(
        discount * product.coupon_rate * interval_years,
        probability_sum,
        discount
    );
}

template<SumStrategy Strategy>
__global__ void range_accrual_kernel(
    const black_scholes::ModelParameters* __restrict__ models,
    const product::RangeAccrualParameters* __restrict__ products,
    std::size_t case_count,
    std::size_t result_count,
    float* __restrict__ prices
) {
    const std::size_t result =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (result < result_count) {
        const std::size_t case_index = result % case_count;
        prices[result] = range_accrual_price<Strategy>(
            models[case_index], products[case_index]
        );
    }
}

template<SumStrategy Strategy>
void launch_strategy(
    const black_scholes::ModelParameters* device_models,
    const product::RangeAccrualParameters* device_products,
    std::size_t case_count,
    std::size_t result_count,
    float* device_prices
) {
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = static_cast<unsigned int>(
        (result_count + threads - 1U) / threads
    );
    range_accrual_kernel<Strategy><<<blocks, threads>>>(
        device_models,
        device_products,
        case_count,
        result_count,
        device_prices
    );
    check_cuda(cudaGetLastError(), "launch range-accrual sum strategy");
}

template<typename Launch>
float measure_milliseconds(Launch&& launch) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create range benchmark start event");
    check_cuda(cudaEventCreate(&stop), "create range benchmark stop event");
    for (int iteration = 0; iteration < 2; ++iteration) launch();
    check_cuda(cudaDeviceSynchronize(), "warm up range benchmark");
    check_cuda(cudaEventRecord(start), "record range benchmark start event");
    for (int iteration = 0; iteration < 8; ++iteration) launch();
    check_cuda(cudaEventRecord(stop), "record range benchmark stop event");
    check_cuda(cudaEventSynchronize(stop), "wait for range benchmark");
    float milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start, stop),
        "measure range benchmark"
    );
    check_cuda(cudaEventDestroy(stop), "destroy range benchmark stop event");
    check_cuda(cudaEventDestroy(start), "destroy range benchmark start event");
    return milliseconds / 8.0f;
}

double normal_cdf(double value) {
    return 0.5 * std::erfc(-value / std::sqrt(2.0));
}

double interval_probability(
    const black_scholes::ModelParameters& model,
    const product::RangeAccrualParameters& product,
    double observation_time
) {
    const double variance = static_cast<double>(model.volatility)
        * static_cast<double>(model.volatility);
    const double mean = std::log(static_cast<double>(model.spot))
        + (static_cast<double>(model.risk_free_rate)
            - static_cast<double>(model.dividend_yield)
            - 0.5 * variance) * observation_time;
    const double deviation =
        static_cast<double>(model.volatility) * std::sqrt(observation_time);
    if (deviation <= 1.0e-14) {
        return mean > std::log(static_cast<double>(product.lower_barrier))
            && mean < std::log(static_cast<double>(product.upper_barrier))
            ? 1.0 : 0.0;
    }
    return normal_cdf(
        (std::log(static_cast<double>(product.upper_barrier)) - mean)
            / deviation
    ) - normal_cdf(
        (std::log(static_cast<double>(product.lower_barrier)) - mean)
            / deviation
    );
}

double reference_price(
    const black_scholes::ModelParameters& model,
    const product::RangeAccrualParameters& product
) {
    constexpr long double day_fraction = 1.0L / 252.0L;
    const std::uint32_t observation_count =
        product.maturity_days / product.observation_interval_days;
    const long double interval_years =
        static_cast<long double>(product.observation_interval_days)
            * day_fraction;
    long double probability_sum = 0.0L;
    for (std::uint32_t observation = 1U;
         observation <= observation_count;
         ++observation) {
        probability_sum += static_cast<long double>(interval_probability(
            model,
            product,
            static_cast<double>(
                static_cast<long double>(observation) * interval_years
            )
        ));
    }
    const long double maturity_years =
        static_cast<long double>(product.maturity_days) * day_fraction;
    const long double discount = std::exp(
        -static_cast<long double>(model.risk_free_rate) * maturity_years
    );
    return static_cast<double>(discount * (
        1.0L + static_cast<long double>(product.coupon_rate)
            * interval_years * probability_sum
    ));
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

        const std::array<black_scholes::ModelParameters, 4U> models = {{
            {1.0f, 0.02f, 0.01f, 0.20f},
            {1.0f, -0.03f, 0.10f, 1.00f},
            {1.0f, 0.12f, 0.0f, 0.01f},
            {1.0f, 0.08f, 0.06f, 0.60f},
        }};
        const std::array<product::RangeAccrualParameters, 4U> products = {{
            {252U, 21U, 0.70f, 1.40f, 0.12f},
            {1764U, 1U, 0.94f, 1.06f, 0.25f},
            {1260U, 5U, 0.25f, 2.50f, 0.02f},
            {1764U, 1U, 0.995f, 1.005f, 0.005f},
        }};
        black_scholes::ModelParameters* device_models = nullptr;
        product::RangeAccrualParameters* device_products = nullptr;
        float* device_prices = nullptr;
        check_cuda(
            cudaMalloc(&device_models, sizeof(models)),
            "allocate range models"
        );
        check_cuda(
            cudaMalloc(&device_products, sizeof(products)),
            "allocate range products"
        );
        check_cuda(
            cudaMemcpy(
                device_models,
                models.data(),
                sizeof(models),
                cudaMemcpyHostToDevice
            ),
            "copy range models"
        );
        check_cuda(
            cudaMemcpy(
                device_products,
                products.data(),
                sizeof(products),
                cudaMemcpyHostToDevice
            ),
            "copy range products"
        );
        check_cuda(
            cudaMalloc(&device_prices, 4U * 4U * sizeof(float)),
            "allocate range numeric prices"
        );
        launch_strategy<SumStrategy::fp64>(
            device_models, device_products, 4U, 4U, device_prices + 0U
        );
        launch_strategy<SumStrategy::fp32>(
            device_models, device_products, 4U, 4U, device_prices + 4U
        );
        launch_strategy<SumStrategy::compensated_fp32>(
            device_models, device_products, 4U, 4U, device_prices + 8U
        );
        launch_strategy<SumStrategy::chunked_fp32>(
            device_models, device_products, 4U, 4U, device_prices + 12U
        );
        std::array<float, 16U> observed{};
        check_cuda(
            cudaMemcpy(
                observed.data(),
                device_prices,
                sizeof(observed),
                cudaMemcpyDeviceToHost
            ),
            "copy range numeric prices"
        );
        double maximum_compensated_error = 0.0;
        for (std::size_t case_index = 0U; case_index < 4U; ++case_index) {
            const double reference = reference_price(
                models[case_index], products[case_index]
            );
            std::cout << "RANGE_NUMERIC case=" << case_index
                      << " observations="
                      << products[case_index].maturity_days
                            / products[case_index].observation_interval_days
                      << " reference=" << std::setprecision(12) << reference;
            constexpr std::array<const char*, 4U> names = {
                "fp64", "fp32", "compensated_fp32", "chunked_fp32"
            };
            for (std::size_t strategy = 0U; strategy < 4U; ++strategy) {
                const double error = relative_error(
                    observed[strategy * 4U + case_index], reference
                );
                std::cout << ' ' << names[strategy] << '='
                          << observed[strategy * 4U + case_index]
                          << ' ' << names[strategy] << "_relative_error="
                          << error;
                if (strategy == 2U) {
                    maximum_compensated_error = std::max(
                        maximum_compensated_error, error
                    );
                }
            }
            std::cout << '\n';
        }
        require(
            maximum_compensated_error < 1.0e-5,
            "Compensated range-accrual price exceeded its reference budget."
        );

        constexpr std::size_t benchmark_rows = 1024U;
        check_cuda(cudaFree(device_prices), "free range numeric prices");
        check_cuda(
            cudaMalloc(&device_prices, benchmark_rows * sizeof(float)),
            "allocate range benchmark prices"
        );
        auto fp64_launch = [&] {
            launch_strategy<SumStrategy::fp64>(
                device_models + 1U, device_products + 1U, 1U,
                benchmark_rows, device_prices
            );
        };
        auto fp32_launch = [&] {
            launch_strategy<SumStrategy::fp32>(
                device_models + 1U, device_products + 1U, 1U,
                benchmark_rows, device_prices
            );
        };
        auto compensated_launch = [&] {
            launch_strategy<SumStrategy::compensated_fp32>(
                device_models + 1U, device_products + 1U, 1U,
                benchmark_rows, device_prices
            );
        };
        auto chunked_launch = [&] {
            launch_strategy<SumStrategy::chunked_fp32>(
                device_models + 1U, device_products + 1U, 1U,
                benchmark_rows, device_prices
            );
        };
        const float fp64_ms = measure_milliseconds(fp64_launch);
        const float fp32_ms = measure_milliseconds(fp32_launch);
        const float compensated_ms = measure_milliseconds(compensated_launch);
        const float chunked_ms = measure_milliseconds(chunked_launch);

        std::vector<double> host_prices(benchmark_rows);
        const auto host_start = std::chrono::steady_clock::now();
        for (std::size_t row = 0U; row < benchmark_rows; ++row) {
            host_prices[row] = reference_price(models[1], products[1]);
        }
        const auto host_stop = std::chrono::steady_clock::now();
        const double host_ms =
            std::chrono::duration<double, std::milli>(host_stop - host_start)
                .count();
        std::cout << "RANGE_PERFORMANCE rows=" << benchmark_rows
                  << " observations=1764"
                  << " fp64_ms=" << fp64_ms
                  << " fp32_ms=" << fp32_ms
                  << " compensated_fp32_ms=" << compensated_ms
                  << " chunked_fp32_ms=" << chunked_ms
                  << " host_ms=" << host_ms << '\n';

        check_cuda(cudaFree(device_prices), "free range benchmark prices");
        check_cuda(cudaFree(device_products), "free range products");
        check_cuda(cudaFree(device_models), "free range models");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
