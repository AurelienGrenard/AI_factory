#include "ai_factory/cuda/black_scholes/api.cuh"

#include "ai_factory/cuda/black_scholes/dynamics.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>

namespace ai_factory::cuda {
namespace {

using rng::standard_normal;
using detail::check_cuda;
using detail::kThreadsPerBlock;
using detail::reusable_cuda_workspace;
using black_scholes_detail::log_step;

constexpr std::size_t kBasisSize = 4U;
constexpr std::size_t kRegressionStatCount = 15U;
constexpr std::size_t kFinalStatCount = 2U;

struct BlackScholesAmericanWorkspaceTag {};

__device__ __forceinline__ double american_put_payoff(
    double spot,
    double strike
) {
    return fmax(strike - spot, 0.0);
}

__device__ __forceinline__ void laguerre_basis(
    double scaled_spot,
    double* basis
) {
    const double x = scaled_spot;
    const double x2 = x * x;
    basis[0] = 1.0;
    basis[1] = 1.0 - x;
    basis[2] = 1.0 - 2.0 * x + 0.5 * x2;
    basis[3] = 1.0 - 3.0 * x + 1.5 * x2 - x2 * x / 6.0;
}

__device__ bool solve_linear_4x4(
    double matrix[kBasisSize][kBasisSize],
    double rhs[kBasisSize],
    double solution[kBasisSize]
) {
    for (std::size_t pivot = 0; pivot < kBasisSize; ++pivot) {
        std::size_t best = pivot;
        double best_abs = fabs(matrix[pivot][pivot]);
        for (std::size_t row = pivot + 1U; row < kBasisSize; ++row) {
            const double candidate = fabs(matrix[row][pivot]);
            if (candidate > best_abs) {
                best = row;
                best_abs = candidate;
            }
        }
        if (best_abs < 1.0e-14) {
            return false;
        }
        if (best != pivot) {
            for (std::size_t column = 0; column < kBasisSize; ++column) {
                const double value = matrix[pivot][column];
                matrix[pivot][column] = matrix[best][column];
                matrix[best][column] = value;
            }
            const double value = rhs[pivot];
            rhs[pivot] = rhs[best];
            rhs[best] = value;
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
    for (std::size_t index = 0; index < kBasisSize; ++index) {
        solution[index] = rhs[index];
    }
    return true;
}

__global__ void american_put_spot_paths_kernel(
    const BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto path =
        static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }

    const auto row = rows[row_index];
    const std::size_t step_count = num_steps + 1U;
    const std::size_t path_offset =
        (row_index * num_paths + path) * step_count;
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    rng::NormalSequence normals(row.seed, 0U, path * num_steps);
    double spot = row.spot;
    spot_paths[path_offset] = spot;
    for (std::size_t step = 0; step < num_steps; ++step) {
        spot *= exp(log_step(row, dt, sqrt_dt, normals.next()));
        spot_paths[path_offset + step + 1U] = spot;
    }
}

__global__ void initialize_lsm_cashflows_kernel(
    const BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    const double* spot_paths,
    double* cashflows,
    int* exercise_steps
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto path =
        static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }
    const auto row = rows[row_index];
    const std::size_t step_count = num_steps + 1U;
    const std::size_t flat_path = row_index * num_paths + path;
    const double terminal = spot_paths[flat_path * step_count + num_steps];
    cashflows[flat_path] = american_put_payoff(terminal, row.strike);
    exercise_steps[flat_path] = static_cast<int>(num_steps);
}

__global__ void regression_partial_sums_kernel(
    const BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t exercise_step,
    const double* spot_paths,
    const double* cashflows,
    const int* exercise_steps,
    double* partials
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    const auto block_path = static_cast<std::size_t>(blockIdx.y);
    if (row_index >= row_count) {
        return;
    }
    extern __shared__ double shared[];
    const auto tid = static_cast<std::size_t>(threadIdx.x);
    double local[kRegressionStatCount]{};
    const auto row = rows[row_index];
    const double dt = row.maturity / static_cast<double>(num_steps);
    const std::size_t step_count = num_steps + 1U;

    for (std::size_t path = block_path * blockDim.x + tid;
         path < num_paths;
         path += static_cast<std::size_t>(gridDim.y) * blockDim.x) {
        const std::size_t flat_path = row_index * num_paths + path;
        const double spot = spot_paths[flat_path * step_count + exercise_step];
        const double immediate = american_put_payoff(spot, row.strike);
        if (immediate <= 0.0) {
            continue;
        }
        double basis[kBasisSize];
        laguerre_basis(spot / row.strike, basis);
        const double target = cashflows[flat_path]
                              * exp(
                                  -row.risk_free_rate * dt
                                  * static_cast<double>(
                                      exercise_steps[flat_path]
                                      - static_cast<int>(exercise_step)
                                  )
                              );
        std::size_t cursor = 0U;
        for (std::size_t i = 0; i < kBasisSize; ++i) {
            for (std::size_t j = i; j < kBasisSize; ++j) {
                local[cursor++] += basis[i] * basis[j];
            }
        }
        for (std::size_t i = 0; i < kBasisSize; ++i) {
            local[cursor++] += basis[i] * target;
        }
        local[cursor] += 1.0;
    }

    for (std::size_t stat = 0; stat < kRegressionStatCount; ++stat) {
        shared[stat * blockDim.x + tid] = local[stat];
    }
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            for (std::size_t stat = 0; stat < kRegressionStatCount; ++stat) {
                shared[stat * blockDim.x + tid] +=
                    shared[stat * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const std::size_t partial_offset =
            (row_index * gridDim.y + block_path) * kRegressionStatCount;
        for (std::size_t stat = 0; stat < kRegressionStatCount; ++stat) {
            partials[partial_offset + stat] = shared[stat * blockDim.x];
        }
    }
}

__global__ void solve_regression_coefficients_kernel(
    const double* partials,
    std::size_t row_count,
    std::size_t partial_count,
    double* coefficients
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count || threadIdx.x != 0) {
        return;
    }
    double matrix[kBasisSize][kBasisSize]{};
    double rhs[kBasisSize]{};
    double itm_count = 0.0;
    for (std::size_t partial = 0; partial < partial_count; ++partial) {
        const double* values =
            partials + (row_index * partial_count + partial) * kRegressionStatCount;
        std::size_t cursor = 0U;
        for (std::size_t i = 0; i < kBasisSize; ++i) {
            for (std::size_t j = i; j < kBasisSize; ++j) {
                const double value = values[cursor++];
                matrix[i][j] += value;
                if (i != j) {
                    matrix[j][i] += value;
                }
            }
        }
        for (std::size_t i = 0; i < kBasisSize; ++i) {
            rhs[i] += values[cursor++];
        }
        itm_count += values[cursor];
    }
    double solution[kBasisSize]{};
    const bool valid = itm_count > static_cast<double>(kBasisSize)
                       && solve_linear_4x4(matrix, rhs, solution);
    const std::size_t offset = row_index * (kBasisSize + 1U);
    for (std::size_t index = 0; index < kBasisSize; ++index) {
        coefficients[offset + index] = valid ? solution[index] : 0.0;
    }
    coefficients[offset + kBasisSize] = valid ? 1.0 : 0.0;
}

__global__ void apply_exercise_decision_kernel(
    const BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    std::size_t exercise_step,
    const double* spot_paths,
    const double* coefficients,
    double* cashflows,
    int* exercise_steps
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto path =
        static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }
    const std::size_t coefficient_offset = row_index * (kBasisSize + 1U);
    if (coefficients[coefficient_offset + kBasisSize] <= 0.0) {
        return;
    }
    const auto row = rows[row_index];
    const std::size_t step_count = num_steps + 1U;
    const std::size_t flat_path = row_index * num_paths + path;
    const double spot = spot_paths[flat_path * step_count + exercise_step];
    const double immediate = american_put_payoff(spot, row.strike);
    if (immediate <= 0.0) {
        return;
    }
    double basis[kBasisSize];
    laguerre_basis(spot / row.strike, basis);
    double continuation = 0.0;
    for (std::size_t index = 0; index < kBasisSize; ++index) {
        continuation += coefficients[coefficient_offset + index] * basis[index];
    }
    if (immediate > continuation) {
        cashflows[flat_path] = immediate;
        exercise_steps[flat_path] = static_cast<int>(exercise_step);
    }
}

__global__ void final_partial_sums_kernel(
    const BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    const double* cashflows,
    const int* exercise_steps,
    double* partials
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    const auto block_path = static_cast<std::size_t>(blockIdx.y);
    if (row_index >= row_count) {
        return;
    }
    extern __shared__ double shared[];
    const auto tid = static_cast<std::size_t>(threadIdx.x);
    double local_sum = 0.0;
    double local_sumsq = 0.0;
    const auto row = rows[row_index];
    const double dt = row.maturity / static_cast<double>(num_steps);

    for (std::size_t path = block_path * blockDim.x + tid;
         path < num_paths;
         path += static_cast<std::size_t>(gridDim.y) * blockDim.x) {
        const std::size_t flat_path = row_index * num_paths + path;
        const double discounted = cashflows[flat_path]
                                  * exp(
                                      -row.risk_free_rate * dt
                                      * static_cast<double>(
                                          exercise_steps[flat_path]
                                      )
                                  );
        local_sum += discounted;
        local_sumsq += discounted * discounted;
    }
    shared[tid] = local_sum;
    shared[blockDim.x + tid] = local_sumsq;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            shared[tid] += shared[tid + stride];
            shared[blockDim.x + tid] += shared[blockDim.x + tid + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const std::size_t offset =
            (row_index * gridDim.y + block_path) * kFinalStatCount;
        partials[offset] = shared[0];
        partials[offset + 1U] = shared[blockDim.x];
    }
}

__global__ void finalize_outputs_kernel(
    const double* partials,
    std::size_t row_count,
    std::size_t partial_count,
    std::size_t num_paths,
    MonteCarloOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count || threadIdx.x != 0) {
        return;
    }
    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t partial = 0; partial < partial_count; ++partial) {
        const auto offset =
            (row_index * partial_count + partial) * kFinalStatCount;
        sum += partials[offset];
        sumsq += partials[offset + 1U];
    }
    const double path_count = static_cast<double>(num_paths);
    const double mean = sum / path_count;
    const double variance =
        (sumsq - path_count * mean * mean) / static_cast<double>(num_paths - 1U);
    outputs[row_index] = {
        mean,
        sqrt(fmax(variance, 0.0)) / sqrt(path_count)
    };
}

}  // namespace

void price_black_scholes_american_put_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    using clock = std::chrono::steady_clock;
    const auto started = clock::now();
    float elapsed_ms = 0.0F;

    const auto path_block_count =
        static_cast<unsigned int>((num_paths + kThreadsPerBlock - 1U)
                                  / kThreadsPerBlock);
    const dim3 path_grid(
        static_cast<unsigned int>(row_count),
        path_block_count,
        1U
    );
    const auto row_bytes = row_count * sizeof(BlackScholesRow);
    const auto path_count = row_count * num_paths;
    const auto spot_count = path_count * (num_steps + 1U);
    const auto regression_partial_count =
        row_count * path_block_count * kRegressionStatCount;
    const auto coefficient_count = row_count * (kBasisSize + 1U);
    const auto final_partial_count =
        row_count * path_block_count * kFinalStatCount;
    const auto output_bytes = row_count * sizeof(MonteCarloOutput);

    auto& workspace = reusable_cuda_workspace<
        BlackScholesAmericanWorkspaceTag,
        8U
    >();
    auto* device_rows = workspace.buffer<BlackScholesRow>(
        0U, row_count, "cudaMalloc american rows"
    );
    auto* device_paths = workspace.buffer<double>(
        1U, spot_count, "cudaMalloc american paths"
    );
    auto* device_cashflows = workspace.buffer<double>(
        2U, path_count, "cudaMalloc american cashflows"
    );
    auto* device_exercise_steps = workspace.buffer<int>(
        3U, path_count, "cudaMalloc american exercise steps"
    );
    auto* device_regression_partials = workspace.buffer<double>(
        4U, regression_partial_count, "cudaMalloc american regression partials"
    );
    auto* device_coefficients = workspace.buffer<double>(
        5U, coefficient_count, "cudaMalloc american coefficients"
    );
    auto* device_final_partials = workspace.buffer<double>(
        6U, final_partial_count, "cudaMalloc american final partials"
    );
    auto* device_outputs = workspace.buffer<MonteCarloOutput>(
        7U, row_count, "cudaMalloc american outputs"
    );
    const auto event_start = workspace.start_event();
    const auto event_stop = workspace.stop_event();

        check_cuda(
            cudaMemcpy(device_rows, host_rows, row_bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy american rows"
        );
        check_cuda(cudaEventRecord(event_start), "cudaEventRecord start");

        american_put_spot_paths_kernel<<<path_grid, kThreadsPerBlock>>>(
            device_rows,
            row_count,
            num_paths,
            num_steps,
            device_paths
        );
        check_cuda(cudaGetLastError(), "black scholes black scholes american put path kernel launch");
        initialize_lsm_cashflows_kernel<<<path_grid, kThreadsPerBlock>>>(
            device_rows,
            row_count,
            num_paths,
            num_steps,
            device_paths,
            device_cashflows,
            device_exercise_steps
        );
        check_cuda(cudaGetLastError(), "black scholes american put init kernel launch");

        const auto regression_shared_bytes =
            kRegressionStatCount * kThreadsPerBlock * sizeof(double);
        for (std::size_t step = num_steps - 1U; step > 0U; --step) {
            regression_partial_sums_kernel<<<
                path_grid,
                kThreadsPerBlock,
                regression_shared_bytes
            >>>(
                device_rows,
                row_count,
                num_paths,
                num_steps,
                step,
                device_paths,
                device_cashflows,
                device_exercise_steps,
                device_regression_partials
            );
            check_cuda(
                cudaGetLastError(),
                "black scholes american put regression partial kernel launch"
            );
            solve_regression_coefficients_kernel<<<
                static_cast<unsigned int>(row_count),
                1U
            >>>(
                device_regression_partials,
                row_count,
                path_block_count,
                device_coefficients
            );
            check_cuda(
                cudaGetLastError(),
                "black scholes american put regression solve kernel launch"
            );
            apply_exercise_decision_kernel<<<path_grid, kThreadsPerBlock>>>(
                device_rows,
                row_count,
                num_paths,
                num_steps,
                step,
                device_paths,
                device_coefficients,
                device_cashflows,
                device_exercise_steps
            );
            check_cuda(
                cudaGetLastError(),
                "black scholes american put exercise kernel launch"
            );
        }

        final_partial_sums_kernel<<<
            path_grid,
            kThreadsPerBlock,
            2U * kThreadsPerBlock * sizeof(double)
        >>>(
            device_rows,
            row_count,
            num_paths,
            num_steps,
            device_cashflows,
            device_exercise_steps,
            device_final_partials
        );
        check_cuda(cudaGetLastError(), "black scholes american put final partial kernel launch");
        finalize_outputs_kernel<<<static_cast<unsigned int>(row_count), 1U>>>(
            device_final_partials,
            row_count,
            path_block_count,
            num_paths,
            device_outputs
        );
        check_cuda(cudaGetLastError(), "black scholes american put final kernel launch");
        check_cuda(cudaEventRecord(event_stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(event_stop), "cudaEventSynchronize stop");
        check_cuda(
            cudaEventElapsedTime(&elapsed_ms, event_start, event_stop),
            "black scholes american put kernel timing"
        );
        check_cuda(
            cudaMemcpy(
                host_outputs,
                device_outputs,
                output_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy american outputs"
        );
    if (timing != nullptr) {
        const auto finished = clock::now();
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(finished - started).count()
        );
    }
}

}  // namespace ai_factory::cuda
