// Qualify the production fractional resolvent against high-precision references.
#include "common/check_cuda.cuh"
#include "common/compensated_sum.cuh"
#include "common/volterra/fractional_resolvent_hybrid_kernel.cuh"

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
using ProductionPolicy = ai_factory::workbench::volterra::
    FractionalResolventHybridKernelPolicy;

struct Fp32FormulaPolicy {
    using Parameters = ProductionPolicy::Parameters;

    struct PreparedKernel {
        Parameters parameters;
        float alpha;
        float laplace_scale;
        float time_step;
        float singular_rough_loading;
        float singular_independent_loading;
    };

    __host__ __device__ __noinline__ static float
    mittag_leffler_alpha_alpha(float alpha, float argument) {
        const float x = -argument;
        // FP32 series cancellation begins much earlier than in FP64.  Use the
        // positive Laplace-density representation beyond order-one arguments.
        const float crossover = 2.0f;
        if (x > crossover) {
            constexpr int points = 96;
            constexpr float pi = 3.14159265358979323846f;
            const float transformed_time = powf(x, 1.0f / alpha);
            const float prefactor = powf(x, 1.0f / alpha - 1.0f)
                / (transformed_time * transformed_time);
            const float sine_scale = sinf(pi * alpha) / pi;
            const float cosine = cosf(pi * alpha);
            CompensatedFloatSum sum;
            #pragma unroll 1
            for (int point = 0; point < points; ++point) {
                const float u = (static_cast<float>(point) + 0.5f)
                    / static_cast<float>(points);
                const float one_minus_u = 1.0f - u;
                const float y = u / one_minus_u;
                const float r = y / transformed_time;
                const float r_to_alpha = powf(r, alpha);
                const float density = sine_scale * powf(r, alpha - 1.0f)
                    / (r_to_alpha * r_to_alpha
                       + 2.0f * r_to_alpha * cosine + 1.0f);
                sum.add(
                    y * expf(-y) * density
                    / (one_minus_u * one_minus_u)
                );
            }
            return prefactor * sum.value() / static_cast<float>(points);
        }
        CompensatedFloatSum sum;
        float power = 1.0f;
        #pragma unroll 1
        for (int order = 0; order < 96; ++order) {
            const float term = power / tgammaf(alpha * order + alpha);
            sum.add(term);
            if (order > 8
                && fabsf(term) < 2.0e-6f * fmaxf(fabsf(sum.value()), 1.0f)) {
                break;
            }
            power *= argument;
        }
        return sum.value();
    }

    __host__ __device__ __noinline__ static float kernel(
        const PreparedKernel& kernel,
        float time_years
    ) {
        const float argument = -kernel.parameters.mean_reversion
            * kernel.laplace_scale * powf(time_years, kernel.alpha);
        return kernel.laplace_scale
            * powf(time_years, kernel.alpha - 1.0f)
            * mittag_leffler_alpha_alpha(kernel.alpha, argument);
    }

    __host__ __device__ __noinline__ static float power_integral(
        const PreparedKernel& kernel,
        float upper,
        int power
    ) {
        if (!(upper > 0.0f)) return 0.0f;
        const float lower_x = -logf(upper);
        const float rate = power == 2
            ? 2.0f * kernel.parameters.hurst_exponent
            : kernel.alpha;
        constexpr int points = 96;
        const float scale = 1.0f / fmaxf(rate, 0.02f);
        CompensatedFloatSum sum;
        #pragma unroll 1
        for (int point = 0; point < points; ++point) {
            const float u = (static_cast<float>(point) + 0.5f)
                / static_cast<float>(points);
            const float one_minus_u = 1.0f - u;
            const float x = lower_x + scale * u / one_minus_u;
            float integrand = 0.0f;
            if (x > 70.0f) {
                const float leading = sqrtf(
                    2.0f * kernel.parameters.hurst_exponent
                );
                integrand = (power == 2 ? leading * leading : leading)
                    * expf(-rate * x);
            } else {
                const float time_years = expf(-x);
                const float value = kernel_fn(kernel, time_years);
                integrand = (power == 2 ? value * value : value)
                    * time_years;
            }
            sum.add(integrand * scale
                / (one_minus_u * one_minus_u));
        }
        return sum.value() / static_cast<float>(points);
    }

    __host__ __device__ __noinline__ static PreparedKernel prepare(
        const Parameters& parameters,
        float time_step
    ) {
        PreparedKernel kernel{
            parameters,
            parameters.hurst_exponent + 0.5f,
            0.0f,
            time_step,
            0.0f,
            0.0f,
        };
        kernel.laplace_scale = sqrtf(2.0f * parameters.hurst_exponent)
            * tgammaf(kernel.alpha);
        const float first_moment = power_integral(kernel, time_step, 1);
        const float second_moment = power_integral(kernel, time_step, 2);
        kernel.singular_rough_loading = first_moment / sqrtf(time_step);
        kernel.singular_independent_loading = sqrtf(fmaxf(
            second_moment
                - kernel.singular_rough_loading
                    * kernel.singular_rough_loading,
            0.0f
        ));
        return kernel;
    }

    __host__ __device__ __noinline__ static float far_cell_weight(
        const PreparedKernel& kernel,
        unsigned int lag
    ) {
        const float lower = static_cast<float>(lag - 1U) * kernel.time_step;
        const float upper = static_cast<float>(lag) * kernel.time_step;
        constexpr int intervals = 8;
        const float h = (upper - lower) / static_cast<float>(intervals);
        CompensatedFloatSum sum;
        sum.add(kernel_fn(kernel, lower));
        sum.add(kernel_fn(kernel, upper));
        for (int index = 1; index < intervals; ++index) {
            sum.add((index & 1 ? 4.0f : 2.0f) * kernel_fn(
                kernel,
                fmaf(static_cast<float>(index), h, lower)
            ));
        }
        return sum.value() * h / (3.0f * kernel.time_step);
    }

private:
    __host__ __device__ __forceinline__ static float kernel_fn(
        const PreparedKernel& kernel,
        float time_years
    ) {
        return Fp32FormulaPolicy::kernel(kernel, time_years);
    }
};

struct Case {
    float hurst;
    float mean_reversion;
    float time_years;
    unsigned int lag;
};

struct Result {
    float production_kernel;
    float formula_kernel;
    float production_first_integral;
    float formula_first_integral;
    float production_second_integral;
    float formula_second_integral;
    float production_far_weight;
    float formula_far_weight;
};

__global__ void evaluate_cases_kernel(
    const Case* __restrict__ cases,
    std::size_t case_count,
    Result* __restrict__ results
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= case_count) return;
    const Case input = cases[index];
    const ProductionPolicy::Parameters parameters{
        input.hurst, input.mean_reversion,
    };
    constexpr float dt = 1.0f / 504.0f;
    const auto production = ProductionPolicy::prepare(parameters, dt);
    const auto formula = Fp32FormulaPolicy::prepare(parameters, dt);
    results[index] = {
        ProductionPolicy::kernel(production, input.time_years),
        Fp32FormulaPolicy::kernel(formula, input.time_years),
        ProductionPolicy::power_integral(production, input.time_years, 1),
        Fp32FormulaPolicy::power_integral(formula, input.time_years, 1),
        ProductionPolicy::power_integral(production, input.time_years, 2),
        Fp32FormulaPolicy::power_integral(formula, input.time_years, 2),
        ProductionPolicy::far_cell_weight(production, input.lag),
        Fp32FormulaPolicy::far_cell_weight(formula, input.lag),
    };
}

template<typename Policy>
__global__ void far_weight_benchmark_kernel(
    typename Policy::PreparedKernel kernel,
    std::size_t row_count,
    float* __restrict__ output
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    float sum = 0.0f;
    for (unsigned int offset = 0U; offset < 16U; ++offset) {
        sum += Policy::far_cell_weight(
            kernel,
            2U + static_cast<unsigned int>((row + offset) % 1007U)
        );
    }
    output[row] = sum;
}

template<typename Policy>
__global__ void integral_benchmark_kernel(
    typename Policy::PreparedKernel kernel,
    std::size_t row_count,
    float* __restrict__ output
) {
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < row_count) {
        const float upper = 0.01f
            + 1.99f * static_cast<float>(row % 257U) / 256.0f;
        output[row] = Policy::power_integral(kernel, upper, 2);
    }
}

template<typename Launch>
float measure_milliseconds(Launch&& launch, int repetitions) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create resolvent benchmark start");
    check_cuda(cudaEventCreate(&stop), "create resolvent benchmark stop");
    launch();
    check_cuda(cudaDeviceSynchronize(), "warm up resolvent benchmark");
    check_cuda(cudaEventRecord(start), "record resolvent benchmark start");
    for (int iteration = 0; iteration < repetitions; ++iteration) launch();
    check_cuda(cudaEventRecord(stop), "record resolvent benchmark stop");
    check_cuda(cudaEventSynchronize(stop), "wait for resolvent benchmark");
    float milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start, stop),
        "measure resolvent benchmark"
    );
    check_cuda(cudaEventDestroy(stop), "destroy resolvent benchmark stop");
    check_cuda(cudaEventDestroy(start), "destroy resolvent benchmark start");
    return milliseconds / static_cast<float>(repetitions);
}

long double reference_mittag(long double alpha, long double argument) {
    const long double x = -argument;
    const long double crossover = 3.5L + 12.0L * (alpha - 0.5L);
    if (x > crossover) {
        constexpr int points = 384;
        const long double pi = acosl(-1.0L);
        const long double transformed_time = powl(x, 1.0L / alpha);
        const long double prefactor = powl(x, 1.0L / alpha - 1.0L)
            / (transformed_time * transformed_time);
        const long double sine_scale = sinl(pi * alpha) / pi;
        const long double cosine = cosl(pi * alpha);
        long double sum = 0.0L;
        for (int point = 0; point < points; ++point) {
            const long double u = (static_cast<long double>(point) + 0.5L)
                / static_cast<long double>(points);
            const long double one_minus_u = 1.0L - u;
            const long double y = u / one_minus_u;
            const long double r = y / transformed_time;
            const long double r_to_alpha = powl(r, alpha);
            const long double density = sine_scale * powl(r, alpha - 1.0L)
                / (r_to_alpha * r_to_alpha
                   + 2.0L * r_to_alpha * cosine + 1.0L);
            sum += y * expl(-y) * density
                / (one_minus_u * one_minus_u);
        }
        return prefactor * sum / static_cast<long double>(points);
    }
    long double sum = 0.0L;
    long double power = 1.0L;
    for (int order = 0; order < 192; ++order) {
        const long double term =
            power / tgammal(alpha * order + alpha);
        sum += term;
        if (order > 12
            && fabsl(term) < 2.0e-18L * std::max(fabsl(sum), 1.0L)) break;
        power *= argument;
    }
    return sum;
}

long double reference_kernel(const Case& input, long double time_years) {
    const long double hurst = input.hurst;
    const long double alpha = hurst + 0.5L;
    const long double scale = sqrtl(2.0L * hurst) * tgammal(alpha);
    const long double argument = -static_cast<long double>(
        input.mean_reversion
    ) * scale * powl(time_years, alpha);
    return scale * powl(time_years, alpha - 1.0L)
        * reference_mittag(alpha, argument);
}

long double reference_integral(const Case& input, int power) {
    constexpr int points = 384;
    const long double upper = input.time_years;
    const long double lower_x = -logl(upper);
    const long double alpha = static_cast<long double>(input.hurst) + 0.5L;
    const long double rate = power == 2
        ? 2.0L * static_cast<long double>(input.hurst)
        : alpha;
    const long double scale = 1.0L / std::max(rate, 0.02L);
    long double sum = 0.0L;
    for (int point = 0; point < points; ++point) {
        const long double u = (static_cast<long double>(point) + 0.5L)
            / static_cast<long double>(points);
        const long double one_minus_u = 1.0L - u;
        const long double x = lower_x + scale * u / one_minus_u;
        long double integrand = 0.0L;
        if (x > 90.0L) {
            const long double leading = sqrtl(
                2.0L * static_cast<long double>(input.hurst)
            );
            integrand = (power == 2 ? leading * leading : leading)
                * expl(-rate * x);
        } else {
            const long double time_years = expl(-x);
            const long double value = reference_kernel(input, time_years);
            integrand = (power == 2 ? value * value : value) * time_years;
        }
        sum += integrand * scale / (one_minus_u * one_minus_u);
    }
    return sum / static_cast<long double>(points);
}

long double reference_far_weight(const Case& input) {
    constexpr long double dt = 1.0L / 504.0L;
    const long double lower = static_cast<long double>(input.lag - 1U) * dt;
    const long double upper = static_cast<long double>(input.lag) * dt;
    constexpr int intervals = 32;
    const long double h = (upper - lower) / intervals;
    long double sum = reference_kernel(input, lower)
        + reference_kernel(input, upper);
    for (int index = 1; index < intervals; ++index) {
        sum += (index & 1 ? 4.0L : 2.0L)
            * reference_kernel(input, lower + index * h);
    }
    return sum * h / (3.0L * dt);
}

double relative_error(double observed, long double reference) {
    return static_cast<double>(fabsl(
        static_cast<long double>(observed) - reference
    ) / std::max(fabsl(reference), 1.0e-18L));
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

        constexpr std::array<float, 5U> hursts = {
            0.01f, 0.03f, 0.10f, 0.25f, 0.45f,
        };
        constexpr std::array<float, 5U> reversions = {
            0.0f, 0.2f, 1.0f, 4.0f, 8.0f,
        };
        constexpr std::array<float, 4U> times = {
            1.0f / 504.0f, 0.05f, 1.0f, 7.0f,
        };
        constexpr std::size_t case_count =
            hursts.size() * reversions.size() * times.size();
        std::array<Case, case_count> cases{};
        std::size_t case_index = 0U;
        for (const float hurst : hursts) {
            for (const float reversion : reversions) {
                for (std::size_t time_index = 0U;
                     time_index < times.size();
                     ++time_index) {
                    cases[case_index++] = {
                        hurst,
                        reversion,
                        times[time_index],
                        static_cast<unsigned int>(
                            std::array<unsigned int, 4U>{2U, 17U, 257U, 1008U}
                                [time_index]
                        ),
                    };
                }
            }
        }

        Case* device_cases = nullptr;
        Result* device_results = nullptr;
        check_cuda(cudaMalloc(&device_cases, sizeof(cases)), "allocate cases");
        check_cuda(
            cudaMalloc(&device_results, case_count * sizeof(Result)),
            "allocate resolvent results"
        );
        check_cuda(
            cudaMemcpy(
                device_cases,
                cases.data(),
                sizeof(cases),
                cudaMemcpyHostToDevice
            ),
            "copy resolvent cases"
        );
        evaluate_cases_kernel<<<1U, 128U>>>(
            device_cases, case_count, device_results
        );
        check_cuda(cudaGetLastError(), "launch resolvent numeric sweep");
        std::array<Result, case_count> results{};
        check_cuda(
            cudaMemcpy(
                results.data(),
                device_results,
                sizeof(results),
                cudaMemcpyDeviceToHost
            ),
            "copy resolvent results"
        );

        double maximum_production_kernel_error = 0.0;
        double maximum_formula_kernel_error = 0.0;
        double maximum_production_integral_error = 0.0;
        double maximum_formula_integral_error = 0.0;
        double maximum_formula_first_integral_error = 0.0;
        double maximum_formula_second_integral_error = 0.0;
        double maximum_production_weight_error = 0.0;
        double maximum_formula_weight_error = 0.0;
        std::size_t worst_formula_kernel = 0U;
        std::size_t worst_formula_first_integral = 0U;
        std::size_t worst_formula_second_integral = 0U;
        std::size_t worst_formula_weight = 0U;
        for (std::size_t index = 0U; index < case_count; ++index) {
            const long double kernel_reference = reference_kernel(
                cases[index], cases[index].time_years
            );
            const long double first_reference = reference_integral(
                cases[index], 1
            );
            const long double second_reference = reference_integral(
                cases[index], 2
            );
            const long double weight_reference = reference_far_weight(
                cases[index]
            );
            maximum_production_kernel_error = std::max(
                maximum_production_kernel_error,
                relative_error(
                    results[index].production_kernel,
                    kernel_reference
                )
            );
            const double formula_kernel_error = relative_error(
                results[index].formula_kernel, kernel_reference
            );
            if (formula_kernel_error > maximum_formula_kernel_error) {
                maximum_formula_kernel_error = formula_kernel_error;
                worst_formula_kernel = index;
            }
            maximum_production_integral_error = std::max({
                maximum_production_integral_error,
                relative_error(
                    results[index].production_first_integral, first_reference
                ),
                relative_error(
                    results[index].production_second_integral, second_reference
                ),
            });
            const double formula_first_integral_error = relative_error(
                results[index].formula_first_integral, first_reference
            );
            if (formula_first_integral_error
                > maximum_formula_first_integral_error) {
                maximum_formula_first_integral_error = formula_first_integral_error;
                worst_formula_first_integral = index;
            }
            const double formula_second_integral_error = relative_error(
                results[index].formula_second_integral, second_reference
            );
            if (formula_second_integral_error
                > maximum_formula_second_integral_error) {
                maximum_formula_second_integral_error =
                    formula_second_integral_error;
                worst_formula_second_integral = index;
            }
            maximum_production_weight_error = std::max(
                maximum_production_weight_error,
                relative_error(
                    results[index].production_far_weight,
                    weight_reference
                )
            );
            const double formula_weight_error = relative_error(
                results[index].formula_far_weight, weight_reference
            );
            if (formula_weight_error > maximum_formula_weight_error) {
                maximum_formula_weight_error = formula_weight_error;
                worst_formula_weight = index;
            }
        }
        maximum_formula_integral_error = std::max(
            maximum_formula_first_integral_error,
            maximum_formula_second_integral_error
        );
        std::cout << std::setprecision(12)
                  << "RESOLVENT_NUMERIC cases=" << case_count
                  << " production_kernel_max_relative_error="
                  << maximum_production_kernel_error
                  << " fp32_formula_kernel_max_relative_error="
                  << maximum_formula_kernel_error
                  << " production_integral_max_relative_error="
                  << maximum_production_integral_error
                  << " fp32_formula_integral_max_relative_error="
                  << maximum_formula_integral_error
                  << " production_weight_max_relative_error="
                  << maximum_production_weight_error
                  << " fp32_formula_weight_max_relative_error="
                  << maximum_formula_weight_error << '\n';
        const auto print_worst = [&](const char* label, std::size_t index) {
            const Case& input = cases[index];
            std::cout << "RESOLVENT_WORST metric=" << label
                      << " H=" << input.hurst
                      << " kappa=" << input.mean_reversion
                      << " time=" << input.time_years
                      << " lag=" << input.lag << '\n';
        };
        print_worst("formula_kernel", worst_formula_kernel);
        print_worst("formula_first_integral", worst_formula_first_integral);
        print_worst("formula_second_integral", worst_formula_second_integral);
        print_worst("formula_weight", worst_formula_weight);
        require(
            maximum_production_kernel_error < 5.0e-4
                && maximum_production_integral_error < 5.0e-4
                && maximum_production_weight_error < 5.0e-4
                && maximum_formula_kernel_error < 5.0e-4
                && maximum_formula_integral_error < 5.0e-4
                && maximum_formula_weight_error < 5.0e-4,
            "Production resolvent exceeded its high-precision error budget."
        );

        constexpr std::size_t benchmark_rows = 1024U;
        float* device_output = nullptr;
        check_cuda(
            cudaMalloc(&device_output, benchmark_rows * sizeof(float)),
            "allocate resolvent benchmark output"
        );
        const ProductionPolicy::Parameters parameters{0.10f, 4.0f};
        constexpr float dt = 1.0f / 504.0f;
        const auto production_prepared = ProductionPolicy::prepare(
            parameters,
            dt
        );
        const auto formula_prepared = Fp32FormulaPolicy::prepare(parameters, dt);
        constexpr unsigned int threads = 128U;
        constexpr unsigned int blocks = static_cast<unsigned int>(
            (benchmark_rows + threads - 1U) / threads
        );
        const float production_weights_ms = measure_milliseconds([&] {
            far_weight_benchmark_kernel<ProductionPolicy><<<blocks, threads>>>(
                production_prepared, benchmark_rows, device_output
            );
            check_cuda(cudaGetLastError(), "launch FP64 weight benchmark");
        }, 4);
        const float formula_weights_ms = measure_milliseconds([&] {
            far_weight_benchmark_kernel<Fp32FormulaPolicy><<<blocks, threads>>>(
                formula_prepared, benchmark_rows, device_output
            );
            check_cuda(cudaGetLastError(), "launch FP32 weight benchmark");
        }, 4);
        const float production_integrals_ms = measure_milliseconds([&] {
            integral_benchmark_kernel<ProductionPolicy><<<blocks, threads>>>(
                production_prepared, benchmark_rows, device_output
            );
            check_cuda(cudaGetLastError(), "launch FP64 integral benchmark");
        }, 4);
        const float formula_integrals_ms = measure_milliseconds([&] {
            integral_benchmark_kernel<Fp32FormulaPolicy><<<blocks, threads>>>(
                formula_prepared, benchmark_rows, device_output
            );
            check_cuda(cudaGetLastError(), "launch FP32 integral benchmark");
        }, 4);
        std::cout << "RESOLVENT_PERFORMANCE rows=" << benchmark_rows
                  << " weights_per_row=16"
                  << " production_weights_ms=" << production_weights_ms
                  << " fp32_formula_weights_ms=" << formula_weights_ms
                  << " production_integrals_ms=" << production_integrals_ms
                  << " fp32_formula_integrals_ms=" << formula_integrals_ms
                  << '\n';

        check_cuda(cudaFree(device_output), "free resolvent benchmark output");
        check_cuda(cudaFree(device_results), "free resolvent results");
        check_cuda(cudaFree(device_cases), "free resolvent cases");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
