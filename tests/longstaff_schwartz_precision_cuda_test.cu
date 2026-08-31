// Qualify FP64 products, solves, predictions and exercise decisions in LSM.
#include "common/check_cuda.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/exercise_decision.cuh"
#include "common/longstaff_schwartz/linear_solver.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
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

namespace lsm = ai_factory::workbench::longstaff_schwartz;
namespace basis = ai_factory::workbench::longstaff_schwartz::basis;
using ai_factory::workbench::check_cuda;

using Regressor =
    lsm::NormalEquationRegressor<basis::LaguerrePolynomialTwoFactorBasis>;

constexpr std::size_t kBasisSize = Regressor::kBasisSize;
constexpr std::size_t kGramValueCount = Regressor::kGramValueCount;
constexpr std::size_t kStatisticCount = kGramValueCount + kBasisSize;

struct RegressionSample {
    float features[kBasisSize];
    float discount;
    float cashflow;
};

struct FormationResult {
    double production[kStatisticCount];
    double fp32_products[kStatisticCount];
};

struct SolverResult {
    double production_coefficients[kBasisSize];
    float fp32_coefficients[kBasisSize];
    bool production_success;
    bool fp32_success;
};

struct DecisionInput {
    float features[kBasisSize];
    float immediate;
    float continued_cashflow;
};

enum class PredictionMode {
    fp64,
    fp32,
    selective,
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

__global__ void compare_formation_kernel(
    const RegressionSample* __restrict__ samples,
    std::size_t sample_count,
    FormationResult* __restrict__ result
) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    FormationResult accumulated{};
    for (std::size_t sample_index = 0U;
         sample_index < sample_count;
         ++sample_index) {
        const RegressionSample& sample = samples[sample_index];
        const double production_target =
            static_cast<double>(sample.discount)
            * static_cast<double>(sample.cashflow);
        const float fp32_target = sample.discount * sample.cashflow;
        std::size_t statistic = 0U;
        for (std::size_t row = 0U; row < kBasisSize; ++row) {
            const double row_value = sample.features[row];
            for (std::size_t column = row;
                 column < kBasisSize;
                 ++column) {
                accumulated.production[statistic] += row_value
                    * static_cast<double>(sample.features[column]);
                accumulated.fp32_products[statistic] +=
                    static_cast<double>(
                        sample.features[row] * sample.features[column]
                    );
                ++statistic;
            }
            accumulated.production[kGramValueCount + row] +=
                row_value * production_target;
            accumulated.fp32_products[kGramValueCount + row] +=
                static_cast<double>(sample.features[row] * fp32_target);
        }
    }
    *result = accumulated;
}

template<bool UseFp32Products>
__global__ void formation_benchmark_kernel(
    std::size_t row_count,
    float* __restrict__ output
) {
    const std::size_t row_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_index >= row_count) return;
    double values[kStatisticCount] = {};
    for (std::size_t sample = 0U; sample < 32U; ++sample) {
        const float u = static_cast<float>((row_index + sample) % 1024U)
            / 1024.0f;
        const float primary = 0.55f + 0.90f * u;
        const float secondary = 0.05f + 0.40f * (1.0f - u) * u;
        const float primary_1 = 1.0f - primary;
        const float features[kBasisSize] = {
            1.0f,
            primary_1,
            fmaf(0.5f * primary, primary, 1.0f - 2.0f * primary),
            secondary,
            secondary * secondary,
            primary_1 * secondary,
        };
        const float discount = 0.90f + 0.09f * u;
        const float cashflow = 0.05f + 1.75f * primary;
        const double production_target = static_cast<double>(discount)
            * static_cast<double>(cashflow);
        const float fp32_target = discount * cashflow;
        std::size_t statistic = 0U;
        #pragma unroll
        for (std::size_t basis_row = 0U;
             basis_row < kBasisSize;
             ++basis_row) {
            #pragma unroll
            for (std::size_t basis_column = basis_row;
                 basis_column < kBasisSize;
                 ++basis_column) {
                if constexpr (UseFp32Products) {
                    values[statistic++] += static_cast<double>(
                        features[basis_row] * features[basis_column]
                    );
                } else {
                    values[statistic++] +=
                        static_cast<double>(features[basis_row])
                        * static_cast<double>(features[basis_column]);
                }
            }
            if constexpr (UseFp32Products) {
                values[kGramValueCount + basis_row] +=
                    static_cast<double>(features[basis_row] * fp32_target);
            } else {
                values[kGramValueCount + basis_row] +=
                    static_cast<double>(features[basis_row])
                    * production_target;
            }
        }
    }
    double checksum = 0.0;
    #pragma unroll
    for (std::size_t statistic = 0U;
         statistic < kStatisticCount;
         ++statistic) {
        checksum += values[statistic];
    }
    output[row_index] = static_cast<float>(checksum);
}

__device__ __forceinline__ bool fp32_cholesky_solve(
    float* gram,
    const float* right_hand_side,
    float* coefficients
) {
    for (std::size_t row = 0U; row < kBasisSize; ++row) {
        for (std::size_t column = 0U; column <= row; ++column) {
            float value = gram[row * kBasisSize + column];
            for (std::size_t inner = 0U; inner < column; ++inner) {
                value = fmaf(
                    -gram[row * kBasisSize + inner],
                    gram[column * kBasisSize + inner],
                    value
                );
            }
            if (row == column) {
                if (!(value > static_cast<float>(
                    lsm::kCholeskyDiagonalFloor
                ))) {
                    return false;
                }
                gram[row * kBasisSize + column] = sqrtf(value);
            } else {
                gram[row * kBasisSize + column] = value
                    / gram[column * kBasisSize + column];
            }
        }
    }
    for (std::size_t row = 0U; row < kBasisSize; ++row) {
        float value = right_hand_side[row];
        for (std::size_t column = 0U; column < row; ++column) {
            value = fmaf(
                -gram[row * kBasisSize + column],
                coefficients[column],
                value
            );
        }
        coefficients[row] = value / gram[row * kBasisSize + row];
    }
    for (std::size_t row = kBasisSize; row-- > 0U;) {
        float value = coefficients[row];
        for (std::size_t column = row + 1U;
             column < kBasisSize;
             ++column) {
            value = fmaf(
                -gram[column * kBasisSize + row],
                coefficients[column],
                value
            );
        }
        coefficients[row] = value / gram[row * kBasisSize + row];
    }
    return true;
}

__global__ void compare_solvers_kernel(
    const double* __restrict__ systems,
    const double* __restrict__ right_hand_sides,
    std::size_t system_count,
    SolverResult* __restrict__ results
) {
    const std::size_t system_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (system_index >= system_count) return;
    double production_gram[kBasisSize * kBasisSize];
    double production_rhs[kBasisSize];
    float fp32_gram[kBasisSize * kBasisSize];
    float fp32_rhs[kBasisSize];
    SolverResult result{};
    for (std::size_t index = 0U;
         index < kBasisSize * kBasisSize;
         ++index) {
        const double value =
            systems[system_index * kBasisSize * kBasisSize + index];
        production_gram[index] = value;
        fp32_gram[index] = static_cast<float>(value);
    }
    for (std::size_t index = 0U; index < kBasisSize; ++index) {
        const double value =
            right_hand_sides[system_index * kBasisSize + index];
        production_rhs[index] = value;
        fp32_rhs[index] = static_cast<float>(value);
    }
    result.production_success = lsm::cholesky_solve_normal_equations(
        production_gram,
        production_rhs,
        result.production_coefficients,
        kBasisSize,
        lsm::kCholeskyDiagonalFloor
    );
    result.fp32_success = fp32_cholesky_solve(
        fp32_gram,
        fp32_rhs,
        result.fp32_coefficients
    );
    results[system_index] = result;
}

template<bool UseFp32>
__global__ void solver_benchmark_kernel(
    std::size_t system_count,
    float* __restrict__ output
) {
    const std::size_t system_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (system_index >= system_count) return;
    if constexpr (UseFp32) {
        float gram[kBasisSize * kBasisSize] = {};
        float rhs[kBasisSize];
        float coefficients[kBasisSize] = {};
        const float perturbation = 1.0e-6f
            * static_cast<float>(system_index % 1024U);
        for (std::size_t row = 0U; row < kBasisSize; ++row) {
            for (std::size_t column = 0U; column < kBasisSize; ++column) {
                gram[row * kBasisSize + column] = row == column
                    ? 1.25f + 0.01f * static_cast<float>(row)
                        + perturbation
                    : 0.02f / static_cast<float>(row + column + 1U);
            }
            rhs[row] = 0.10f * static_cast<float>(row + 1U)
                + perturbation;
        }
        const bool success = fp32_cholesky_solve(gram, rhs, coefficients);
        output[system_index] = success ? coefficients[0] : NAN;
    } else {
        double gram[kBasisSize * kBasisSize] = {};
        double rhs[kBasisSize];
        double coefficients[kBasisSize] = {};
        const double perturbation = 1.0e-6
            * static_cast<double>(system_index % 1024U);
        for (std::size_t row = 0U; row < kBasisSize; ++row) {
            for (std::size_t column = 0U; column < kBasisSize; ++column) {
                gram[row * kBasisSize + column] = row == column
                    ? 1.25 + 0.01 * static_cast<double>(row)
                        + perturbation
                    : 0.02 / static_cast<double>(row + column + 1U);
            }
            rhs[row] = 0.10 * static_cast<double>(row + 1U)
                + perturbation;
        }
        const bool success = lsm::cholesky_solve_normal_equations(
            gram,
            rhs,
            coefficients,
            kBasisSize,
            lsm::kCholeskyDiagonalFloor
        );
        output[system_index] = success
            ? static_cast<float>(coefficients[0])
            : NAN;
    }
}

__device__ __forceinline__ double fp64_prediction(
    const DecisionInput& input,
    const double* coefficients
) {
    Regressor::Features features{};
    #pragma unroll
    for (std::size_t index = 0U; index < kBasisSize; ++index) {
        features.values[index] = input.features[index];
    }
    return Regressor::predict(features, coefficients);
}

__device__ __forceinline__ float fp32_prediction(
    const DecisionInput& input,
    const double* coefficients
) {
    float continuation = 0.0f;
    #pragma unroll
    for (std::size_t index = 0U; index < kBasisSize; ++index) {
        continuation = fmaf(
            static_cast<float>(coefficients[index]),
            input.features[index],
            continuation
        );
    }
    return continuation;
}

template<PredictionMode Mode>
__global__ void prediction_benchmark_kernel(
    const DecisionInput* __restrict__ inputs,
    std::size_t input_count,
    const double* __restrict__ coefficients,
    float* __restrict__ updated_cashflows,
    unsigned char* __restrict__ selective_fallbacks
) {
    const std::size_t input_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (input_index >= input_count) return;
    const DecisionInput input = inputs[input_index];
    if constexpr (Mode == PredictionMode::fp64) {
        updated_cashflows[input_index] = lsm::select_exercise_cashflow(
            input.immediate,
            fp64_prediction(input, coefficients),
            input.continued_cashflow
        );
    } else if constexpr (Mode == PredictionMode::fp32) {
        updated_cashflows[input_index] =
            input.immediate > fp32_prediction(input, coefficients)
            ? input.immediate
            : input.continued_cashflow;
    } else {
        const float approximate = fp32_prediction(input, coefficients);
        double coefficient_cast_error = 0.0;
        double absolute_sum = 0.0;
        #pragma unroll
        for (std::size_t index = 0U; index < kBasisSize; ++index) {
            const double feature = input.features[index];
            const float narrowed = static_cast<float>(coefficients[index]);
            coefficient_cast_error += fabs(
                coefficients[index] - static_cast<double>(narrowed)
            ) * fabs(feature);
            absolute_sum += fabs(
                static_cast<double>(narrowed) * feature
            );
        }
        constexpr double gamma =
            (kBasisSize * static_cast<double>(FLT_EPSILON))
            / (1.0 - kBasisSize * static_cast<double>(FLT_EPSILON));
        const double error_bound = coefficient_cast_error
            + gamma * absolute_sum;
        const bool fallback = fabs(
            static_cast<double>(input.immediate)
            - static_cast<double>(approximate)
        ) <= error_bound;
        const double continuation = fallback
            ? fp64_prediction(input, coefficients)
            : static_cast<double>(approximate);
        updated_cashflows[input_index] = lsm::select_exercise_cashflow(
            input.immediate,
            continuation,
            input.continued_cashflow
        );
        selective_fallbacks[input_index] = fallback ? 1U : 0U;
    }
}

template<typename Launch>
float measure_milliseconds(Launch&& launch, int repetitions) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "create LSM benchmark start");
    check_cuda(cudaEventCreate(&stop), "create LSM benchmark stop");
    launch();
    check_cuda(cudaDeviceSynchronize(), "warm up LSM benchmark");
    check_cuda(cudaEventRecord(start), "record LSM benchmark start");
    for (int iteration = 0; iteration < repetitions; ++iteration) launch();
    check_cuda(cudaEventRecord(stop), "record LSM benchmark stop");
    check_cuda(cudaEventSynchronize(stop), "wait for LSM benchmark");
    float milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&milliseconds, start, stop),
        "measure LSM benchmark"
    );
    check_cuda(cudaEventDestroy(stop), "destroy LSM benchmark stop");
    check_cuda(cudaEventDestroy(start), "destroy LSM benchmark start");
    return milliseconds / static_cast<float>(repetitions);
}

std::array<float, kBasisSize> make_features(float primary, float secondary) {
    const float primary_1 = 1.0f - primary;
    return {
        1.0f,
        primary_1,
        std::fma(0.5f * primary, primary, 1.0f - 2.0f * primary),
        secondary,
        secondary * secondary,
        primary_1 * secondary,
    };
}

bool solve_high_precision(
    std::array<long double, kBasisSize * kBasisSize> matrix,
    std::array<long double, kBasisSize> right_hand_side,
    std::array<long double, kBasisSize>& coefficients
) {
    for (std::size_t column = 0U; column < kBasisSize; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1U; row < kBasisSize; ++row) {
            if (fabsl(matrix[row * kBasisSize + column])
                > fabsl(matrix[pivot * kBasisSize + column])) {
                pivot = row;
            }
        }
        if (!(fabsl(matrix[pivot * kBasisSize + column]) > 1.0e-24L)) {
            return false;
        }
        if (pivot != column) {
            for (std::size_t index = column;
                 index < kBasisSize;
                 ++index) {
                std::swap(
                    matrix[column * kBasisSize + index],
                    matrix[pivot * kBasisSize + index]
                );
            }
            std::swap(right_hand_side[column], right_hand_side[pivot]);
        }
        for (std::size_t row = column + 1U; row < kBasisSize; ++row) {
            const long double factor =
                matrix[row * kBasisSize + column]
                / matrix[column * kBasisSize + column];
            for (std::size_t index = column;
                 index < kBasisSize;
                 ++index) {
                matrix[row * kBasisSize + index] -= factor
                    * matrix[column * kBasisSize + index];
            }
            right_hand_side[row] -= factor * right_hand_side[column];
        }
    }
    for (std::size_t row = kBasisSize; row-- > 0U;) {
        long double value = right_hand_side[row];
        for (std::size_t column = row + 1U;
             column < kBasisSize;
             ++column) {
            value -= matrix[row * kBasisSize + column]
                * coefficients[column];
        }
        coefficients[row] = value / matrix[row * kBasisSize + row];
    }
    return true;
}

double relative_error(double observed, long double reference) {
    return static_cast<double>(fabsl(
        static_cast<long double>(observed) - reference
    ) / std::max(fabsl(reference), 1.0e-30L));
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        const cudaError_t availability = cudaGetDeviceCount(&device_count);
        if (availability == cudaErrorNoDevice || device_count == 0) return 77;
        check_cuda(availability, "query CUDA devices for LSM precision");

        constexpr std::size_t formation_sample_count = 32'768U;
        std::vector<RegressionSample> samples(formation_sample_count);
        std::array<long double, kStatisticCount> reference_statistics{};
        for (std::size_t index = 0U;
             index < formation_sample_count;
             ++index) {
            const float u = (static_cast<float>(index) + 0.5f)
                / static_cast<float>(formation_sample_count);
            const float primary = index < formation_sample_count / 2U
                ? 0.55f + 0.90f * u
                : 1.0f + 2.0e-3f * (u - 0.75f);
            const float secondary = index < formation_sample_count / 2U
                ? 0.05f + 0.40f * u * (1.0f - u)
                : 0.25f * primary
                    + 2.0e-4f * std::sin(37.0f * u);
            const auto features = make_features(primary, secondary);
            RegressionSample& sample = samples[index];
            std::copy(features.begin(), features.end(), sample.features);
            sample.discount = 0.87f + 0.12f * u;
            sample.cashflow = 0.02f + 2.5f * primary
                + 0.1f * std::cos(29.0f * u);
            const long double target =
                static_cast<long double>(sample.discount)
                * static_cast<long double>(sample.cashflow);
            std::size_t statistic = 0U;
            for (std::size_t row = 0U; row < kBasisSize; ++row) {
                for (std::size_t column = row;
                     column < kBasisSize;
                     ++column) {
                    reference_statistics[statistic++] +=
                        static_cast<long double>(sample.features[row])
                        * static_cast<long double>(sample.features[column]);
                }
                reference_statistics[kGramValueCount + row] +=
                    static_cast<long double>(sample.features[row]) * target;
            }
        }

        RegressionSample* device_samples = nullptr;
        FormationResult* device_formation = nullptr;
        check_cuda(
            cudaMalloc(
                &device_samples,
                samples.size() * sizeof(RegressionSample)
            ),
            "allocate LSM formation samples"
        );
        check_cuda(
            cudaMalloc(&device_formation, sizeof(FormationResult)),
            "allocate LSM formation result"
        );
        check_cuda(
            cudaMemcpy(
                device_samples,
                samples.data(),
                samples.size() * sizeof(RegressionSample),
                cudaMemcpyHostToDevice
            ),
            "copy LSM formation samples"
        );
        compare_formation_kernel<<<1U, 1U>>>(
            device_samples,
            samples.size(),
            device_formation
        );
        check_cuda(cudaGetLastError(), "launch LSM formation comparison");
        FormationResult formation{};
        check_cuda(
            cudaMemcpy(
                &formation,
                device_formation,
                sizeof(formation),
                cudaMemcpyDeviceToHost
            ),
            "copy LSM formation result"
        );
        double production_formation_error = 0.0;
        double fp32_formation_error = 0.0;
        for (std::size_t index = 0U; index < kStatisticCount; ++index) {
            production_formation_error = std::max(
                production_formation_error,
                relative_error(
                    formation.production[index],
                    reference_statistics[index]
                )
            );
            fp32_formation_error = std::max(
                fp32_formation_error,
                relative_error(
                    formation.fp32_products[index],
                    reference_statistics[index]
                )
            );
        }
        require(
            production_formation_error < 2.0e-12,
            "FP64 LSM products exceeded the high-precision budget."
        );
        require(
            fp32_formation_error > 100.0 * production_formation_error,
            "The FP32 product alternative did not expose precision loss."
        );

        constexpr std::size_t formation_rows = 16'384U;
        float* device_benchmark_output = nullptr;
        check_cuda(
            cudaMalloc(
                &device_benchmark_output,
                formation_rows * sizeof(float)
            ),
            "allocate LSM benchmark output"
        );
        constexpr unsigned int threads = 128U;
        constexpr unsigned int formation_blocks = static_cast<unsigned int>(
            (formation_rows + threads - 1U) / threads
        );
        const float production_formation_ms = measure_milliseconds([&] {
            formation_benchmark_kernel<false>
                <<<formation_blocks, threads>>>(
                    formation_rows,
                    device_benchmark_output
                );
            check_cuda(cudaGetLastError(), "launch FP64 product benchmark");
        }, 6);
        const float fp32_formation_ms = measure_milliseconds([&] {
            formation_benchmark_kernel<true>
                <<<formation_blocks, threads>>>(
                    formation_rows,
                    device_benchmark_output
                );
            check_cuda(cudaGetLastError(), "launch FP32 product benchmark");
        }, 6);

        constexpr std::size_t system_count = 2U;
        std::array<double, system_count * kBasisSize * kBasisSize> systems{};
        std::array<double, system_count * kBasisSize> right_hand_sides{};
        const std::array<double, kBasisSize> desired_coefficients = {
            0.75, -0.40, 0.20, 0.10, -0.05, 0.025,
        };
        for (std::size_t system = 0U; system < system_count; ++system) {
            double* const gram = systems.data()
                + system * kBasisSize * kBasisSize;
            for (std::size_t row = 0U; row < kBasisSize; ++row) {
                for (std::size_t column = 0U;
                     column < kBasisSize;
                     ++column) {
                    gram[row * kBasisSize + column] = row == column
                        ? 1.0 + 0.1 * static_cast<double>(row)
                        : 0.01 / static_cast<double>(row + column + 1U);
                }
            }
            if (system == 1U) {
                constexpr double separation = 2.0e-8;
                for (std::size_t row = 0U; row < kBasisSize; ++row) {
                    for (std::size_t column = 0U;
                         column < kBasisSize;
                         ++column) {
                        if (row != column) {
                            gram[row * kBasisSize + column] = 0.0;
                        }
                    }
                }
                gram[0U] = 1.0;
                gram[1U] = 1.0 - separation;
                gram[kBasisSize] = 1.0 - separation;
                gram[kBasisSize + 1U] = 1.0;
            }
            double trace = 0.0;
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                trace += gram[index * kBasisSize + index];
            }
            const double ridge = lsm::kRidgeRelative * trace
                / static_cast<double>(kBasisSize);
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                gram[index * kBasisSize + index] += ridge;
            }
            for (std::size_t row = 0U; row < kBasisSize; ++row) {
                double rhs = 0.0;
                for (std::size_t column = 0U;
                     column < kBasisSize;
                     ++column) {
                    rhs += gram[row * kBasisSize + column]
                        * desired_coefficients[column];
                }
                right_hand_sides[system * kBasisSize + row] = rhs;
            }
        }

        double* device_systems = nullptr;
        double* device_right_hand_sides = nullptr;
        SolverResult* device_solver_results = nullptr;
        check_cuda(
            cudaMalloc(&device_systems, sizeof(systems)),
            "allocate LSM systems"
        );
        check_cuda(
            cudaMalloc(&device_right_hand_sides, sizeof(right_hand_sides)),
            "allocate LSM right-hand sides"
        );
        check_cuda(
            cudaMalloc(
                &device_solver_results,
                system_count * sizeof(SolverResult)
            ),
            "allocate LSM solver results"
        );
        check_cuda(
            cudaMemcpy(
                device_systems,
                systems.data(),
                sizeof(systems),
                cudaMemcpyHostToDevice
            ),
            "copy LSM systems"
        );
        check_cuda(
            cudaMemcpy(
                device_right_hand_sides,
                right_hand_sides.data(),
                sizeof(right_hand_sides),
                cudaMemcpyHostToDevice
            ),
            "copy LSM right-hand sides"
        );
        compare_solvers_kernel<<<1U, 32U>>>(
            device_systems,
            device_right_hand_sides,
            system_count,
            device_solver_results
        );
        check_cuda(cudaGetLastError(), "launch LSM solver comparison");
        std::array<SolverResult, system_count> solver_results{};
        check_cuda(
            cudaMemcpy(
                solver_results.data(),
                device_solver_results,
                sizeof(solver_results),
                cudaMemcpyDeviceToHost
            ),
            "copy LSM solver results"
        );
        double maximum_production_solver_error = 0.0;
        double maximum_fp32_solver_error = 0.0;
        for (std::size_t system = 0U; system < system_count; ++system) {
            std::array<long double, kBasisSize * kBasisSize> matrix{};
            std::array<long double, kBasisSize> rhs{};
            std::array<long double, kBasisSize> reference{};
            for (std::size_t index = 0U; index < matrix.size(); ++index) {
                matrix[index] = systems[
                    system * kBasisSize * kBasisSize + index
                ];
            }
            for (std::size_t index = 0U; index < rhs.size(); ++index) {
                rhs[index] = right_hand_sides[system * kBasisSize + index];
            }
            require(
                solve_high_precision(matrix, rhs, reference),
                "High-precision LSM reference solve failed."
            );
            require(
                solver_results[system].production_success,
                "FP64 LSM solve failed on a qualified system."
            );
            for (std::size_t index = 0U; index < kBasisSize; ++index) {
                maximum_production_solver_error = std::max(
                    maximum_production_solver_error,
                    relative_error(
                        solver_results[system]
                            .production_coefficients[index],
                        reference[index]
                    )
                );
                if (solver_results[system].fp32_success) {
                    maximum_fp32_solver_error = std::max(
                        maximum_fp32_solver_error,
                        relative_error(
                            solver_results[system].fp32_coefficients[index],
                            reference[index]
                        )
                    );
                } else {
                    maximum_fp32_solver_error =
                        std::numeric_limits<double>::infinity();
                }
            }
        }
        require(
            maximum_production_solver_error < 2.0e-7,
            "FP64 Cholesky exceeded its high-precision coefficient budget."
        );
        require(
            !std::isfinite(maximum_fp32_solver_error)
                || maximum_fp32_solver_error > 1.0e-3,
            "FP32 Cholesky did not expose its stress-system failure."
        );

        constexpr std::size_t solver_benchmark_count = 65'536U;
        check_cuda(cudaFree(device_benchmark_output), "resize benchmark output");
        check_cuda(
            cudaMalloc(
                &device_benchmark_output,
                solver_benchmark_count * sizeof(float)
            ),
            "allocate solver benchmark output"
        );
        constexpr unsigned int solver_blocks = static_cast<unsigned int>(
            (solver_benchmark_count + threads - 1U) / threads
        );
        const float production_solver_ms = measure_milliseconds([&] {
            solver_benchmark_kernel<false><<<solver_blocks, threads>>>(
                solver_benchmark_count,
                device_benchmark_output
            );
            check_cuda(cudaGetLastError(), "launch FP64 solver benchmark");
        }, 8);
        const float fp32_solver_ms = measure_milliseconds([&] {
            solver_benchmark_kernel<true><<<solver_blocks, threads>>>(
                solver_benchmark_count,
                device_benchmark_output
            );
            check_cuda(cudaGetLastError(), "launch FP32 solver benchmark");
        }, 8);

        constexpr std::size_t decision_count = 1U << 20U;
        const std::array<double, kBasisSize> prediction_coefficients = {
            0.875 + 0x1p-30,
            0.125 - 0x1p-29,
            -0.050 + 0x1p-28,
            0.080 - 0x1p-30,
            -0.025 + 0x1p-29,
            0.040 - 0x1p-28,
        };
        std::vector<DecisionInput> decisions(decision_count);
        std::size_t margins_below_1e7 = 0U;
        std::size_t margins_below_1e5 = 0U;
        std::size_t margins_below_1e3 = 0U;
        for (std::size_t index = 0U; index < decision_count; ++index) {
            const float u = static_cast<float>(index % 8192U) / 8192.0f;
            const auto features = make_features(
                0.65f + 0.70f * u,
                0.05f + 0.30f * u * (1.0f - u)
            );
            DecisionInput& input = decisions[index];
            std::copy(features.begin(), features.end(), input.features);
            double continuation = 0.0;
            for (std::size_t feature = 0U;
                 feature < kBasisSize;
                 ++feature) {
                continuation = std::fma(
                    prediction_coefficients[feature],
                    static_cast<double>(input.features[feature]),
                    continuation
                );
            }
            input.immediate = index % 64U == 0U
                ? static_cast<float>(continuation)
                : static_cast<float>(
                    continuation + (index & 1U ? 0.02 : -0.02)
                );
            input.continued_cashflow = 0.20f + 0.10f * u;
            const double margin = fabs(
                static_cast<double>(input.immediate) - continuation
            );
            margins_below_1e7 += margin <= 1.0e-7;
            margins_below_1e5 += margin <= 1.0e-5;
            margins_below_1e3 += margin <= 1.0e-3;
        }

        DecisionInput* device_decisions = nullptr;
        double* device_prediction_coefficients = nullptr;
        float* device_fp64_cashflows = nullptr;
        float* device_fp32_cashflows = nullptr;
        float* device_selective_cashflows = nullptr;
        unsigned char* device_fallbacks = nullptr;
        check_cuda(
            cudaMalloc(
                &device_decisions,
                decisions.size() * sizeof(DecisionInput)
            ),
            "allocate LSM decision inputs"
        );
        check_cuda(
            cudaMalloc(
                &device_prediction_coefficients,
                sizeof(prediction_coefficients)
            ),
            "allocate LSM prediction coefficients"
        );
        check_cuda(
            cudaMalloc(&device_fp64_cashflows, decision_count * sizeof(float)),
            "allocate FP64 LSM cashflows"
        );
        check_cuda(
            cudaMalloc(&device_fp32_cashflows, decision_count * sizeof(float)),
            "allocate FP32 LSM cashflows"
        );
        check_cuda(
            cudaMalloc(
                &device_selective_cashflows,
                decision_count * sizeof(float)
            ),
            "allocate selective LSM cashflows"
        );
        check_cuda(
            cudaMalloc(&device_fallbacks, decision_count),
            "allocate selective LSM fallbacks"
        );
        check_cuda(
            cudaMemcpy(
                device_decisions,
                decisions.data(),
                decisions.size() * sizeof(DecisionInput),
                cudaMemcpyHostToDevice
            ),
            "copy LSM decision inputs"
        );
        check_cuda(
            cudaMemcpy(
                device_prediction_coefficients,
                prediction_coefficients.data(),
                sizeof(prediction_coefficients),
                cudaMemcpyHostToDevice
            ),
            "copy LSM prediction coefficients"
        );
        constexpr unsigned int decision_blocks = static_cast<unsigned int>(
            (decision_count + threads - 1U) / threads
        );
        const float fp64_prediction_ms = measure_milliseconds([&] {
            prediction_benchmark_kernel<PredictionMode::fp64>
                <<<decision_blocks, threads>>>(
                    device_decisions,
                    decision_count,
                    device_prediction_coefficients,
                    device_fp64_cashflows,
                    nullptr
                );
            check_cuda(cudaGetLastError(), "launch FP64 prediction benchmark");
        }, 8);
        const float fp32_prediction_ms = measure_milliseconds([&] {
            prediction_benchmark_kernel<PredictionMode::fp32>
                <<<decision_blocks, threads>>>(
                    device_decisions,
                    decision_count,
                    device_prediction_coefficients,
                    device_fp32_cashflows,
                    nullptr
                );
            check_cuda(cudaGetLastError(), "launch FP32 prediction benchmark");
        }, 8);
        const float selective_prediction_ms = measure_milliseconds([&] {
            prediction_benchmark_kernel<PredictionMode::selective>
                <<<decision_blocks, threads>>>(
                    device_decisions,
                    decision_count,
                    device_prediction_coefficients,
                    device_selective_cashflows,
                    device_fallbacks
                );
            check_cuda(
                cudaGetLastError(),
                "launch selective prediction benchmark"
            );
        }, 8);

        std::vector<float> fp64_cashflows(decision_count);
        std::vector<float> fp32_cashflows(decision_count);
        std::vector<float> selective_cashflows(decision_count);
        std::vector<unsigned char> fallbacks(decision_count);
        check_cuda(
            cudaMemcpy(
                fp64_cashflows.data(),
                device_fp64_cashflows,
                decision_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "copy FP64 LSM cashflows"
        );
        check_cuda(
            cudaMemcpy(
                fp32_cashflows.data(),
                device_fp32_cashflows,
                decision_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "copy FP32 LSM cashflows"
        );
        check_cuda(
            cudaMemcpy(
                selective_cashflows.data(),
                device_selective_cashflows,
                decision_count * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "copy selective LSM cashflows"
        );
        check_cuda(
            cudaMemcpy(
                fallbacks.data(),
                device_fallbacks,
                decision_count,
                cudaMemcpyDeviceToHost
            ),
            "copy selective LSM fallbacks"
        );
        std::size_t fp32_decision_divergences = 0U;
        std::size_t selective_decision_divergences = 0U;
        std::size_t selective_fallback_count = 0U;
        long double fp64_price = 0.0L;
        long double fp32_price = 0.0L;
        long double selective_price = 0.0L;
        for (std::size_t index = 0U; index < decision_count; ++index) {
            fp32_decision_divergences +=
                fp32_cashflows[index] != fp64_cashflows[index];
            selective_decision_divergences +=
                selective_cashflows[index] != fp64_cashflows[index];
            selective_fallback_count += fallbacks[index] != 0U;
            fp64_price += fp64_cashflows[index];
            fp32_price += fp32_cashflows[index];
            selective_price += selective_cashflows[index];
        }
        fp64_price /= static_cast<long double>(decision_count);
        fp32_price /= static_cast<long double>(decision_count);
        selective_price /= static_cast<long double>(decision_count);
        require(
            fp32_decision_divergences > 0U,
            "FP32 LSM prediction did not expose a boundary divergence."
        );
        require(
            selective_decision_divergences == 0U,
            "Selective LSM prediction failed to preserve an FP64 decision."
        );

        std::cout << std::setprecision(12)
                  << "LSM_FORMATION samples=" << formation_sample_count
                  << " production_max_relative_error="
                  << production_formation_error
                  << " fp32_product_max_relative_error="
                  << fp32_formation_error
                  << " production_ms=" << production_formation_ms
                  << " fp32_product_ms=" << fp32_formation_ms << '\n'
                  << "LSM_SOLVER systems=" << system_count
                  << " stress_condition_estimate=1e8"
                  << " production_max_relative_error="
                  << maximum_production_solver_error
                  << " fp32_max_relative_error="
                  << maximum_fp32_solver_error
                  << " production_65536_ms=" << production_solver_ms
                  << " fp32_65536_ms=" << fp32_solver_ms << '\n'
                  << "LSM_DECISION cases=" << decision_count
                  << " margin_le_1e-7=" << margins_below_1e7
                  << " margin_le_1e-5=" << margins_below_1e5
                  << " margin_le_1e-3=" << margins_below_1e3
                  << " fp32_divergences=" << fp32_decision_divergences
                  << " selective_divergences="
                  << selective_decision_divergences
                  << " selective_fallbacks=" << selective_fallback_count
                  << " fp64_price=" << static_cast<double>(fp64_price)
                  << " fp32_price=" << static_cast<double>(fp32_price)
                  << " selective_price="
                  << static_cast<double>(selective_price)
                  << " fp64_ms=" << fp64_prediction_ms
                  << " fp32_ms=" << fp32_prediction_ms
                  << " selective_ms=" << selective_prediction_ms << '\n';

        check_cuda(cudaFree(device_fallbacks), "free selective fallbacks");
        check_cuda(
            cudaFree(device_selective_cashflows),
            "free selective cashflows"
        );
        check_cuda(cudaFree(device_fp32_cashflows), "free FP32 cashflows");
        check_cuda(cudaFree(device_fp64_cashflows), "free FP64 cashflows");
        check_cuda(
            cudaFree(device_prediction_coefficients),
            "free prediction coefficients"
        );
        check_cuda(cudaFree(device_decisions), "free decision inputs");
        check_cuda(cudaFree(device_benchmark_output), "free benchmark output");
        check_cuda(cudaFree(device_solver_results), "free solver results");
        check_cuda(cudaFree(device_right_hand_sides), "free right-hand sides");
        check_cuda(cudaFree(device_systems), "free LSM systems");
        check_cuda(cudaFree(device_formation), "free formation result");
        check_cuda(cudaFree(device_samples), "free formation samples");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
