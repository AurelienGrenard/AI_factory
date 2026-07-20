#include "ai_factory/cuda/heston/api.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::cuda {
namespace {

using rng::standard_normal;
using rng::standard_normal_pair;
using rng::standard_uniform;
using detail::check_cuda;
constexpr int kThreadsPerBlock = 128;
using heston_detail::advance_heston_qe_step;
using heston_detail::advance_heston_qe_step_precomputed;
using heston_detail::advance_heston_step;
using heston_detail::kHestonEulerFullTruncation;
using heston_detail::make_qe_coefficients;

__global__ void heston_terminal_spots_kernel(
    const HestonRow* row_ptr,
    std::size_t num_paths,
    std::size_t num_steps,
    double* terminal_spots
) {
    const auto path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }

    const auto row = *row_ptr;
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift_scale = row.risk_free_rate - row.dividend_yield;
    const double correlation_scale = sqrt(1.0 - row.rho * row.rho);
    const double kappa_dt = row.kappa * dt;
    const double vol_of_var_sqrt_dt = row.volatility_of_variance * sqrt_dt;

    double spot = row.spot;
    double variance = row.initial_variance;
    if (row.scheme == kHestonEulerFullTruncation) {
        std::size_t step = 0;
        for (; step + 1U < num_steps;) {
            const auto normal_index = path * num_steps + step;
            if ((normal_index % 2ULL) != 0ULL) {
                break;
            }
            const auto pair_index = normal_index / 2ULL;
            const auto spot_pair = standard_normal_pair(row.seed, 0, pair_index);
            const auto variance_pair =
                standard_normal_pair(row.seed, 1, pair_index);
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                spot_pair.first,
                variance_pair.first,
                spot,
                variance
            );
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                spot_pair.second,
                variance_pair.second,
                spot,
                variance
            );
            step += 2U;
        }
        for (; step < num_steps; ++step) {
            const auto normal_index = path * num_steps + step;
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                standard_normal(row.seed, 0, normal_index),
                standard_normal(row.seed, 1, normal_index),
                spot,
                variance
            );
        }
    } else {
        double log_spot = log(row.spot);
        const auto coefficients = make_qe_coefficients(row, dt);
        const auto first_index = path * num_steps;
        rng::NormalSequence variance_normals(row.seed, 0U, first_index);
        rng::UniformSequence variance_uniforms(row.seed, 2U, first_index);
        rng::NormalSequence independent_normals(row.seed, 1U, first_index);
        for (std::size_t step = 0; step < num_steps; ++step) {
            advance_heston_qe_step_precomputed(
                row,
                coefficients,
                variance_normals.next(),
                variance_uniforms.next(),
                independent_normals.next(),
                log_spot,
                variance
            );
        }
        spot = exp(log_spot);
    }
    terminal_spots[path] = spot;
}

__global__ void heston_max_spots_kernel(
    const HestonRow* row_ptr,
    std::size_t num_paths,
    std::size_t num_steps,
    double* max_spots
) {
    const auto path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }

    const auto row = *row_ptr;
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift_scale = row.risk_free_rate - row.dividend_yield;
    const double correlation_scale = sqrt(1.0 - row.rho * row.rho);
    const double kappa_dt = row.kappa * dt;
    const double vol_of_var_sqrt_dt = row.volatility_of_variance * sqrt_dt;

    double spot = row.spot;
    double max_spot = row.spot;
    double variance = row.initial_variance;
    if (row.scheme == kHestonEulerFullTruncation) {
        std::size_t step = 0;
        for (; step + 1U < num_steps;) {
            const auto normal_index = path * num_steps + step;
            if ((normal_index % 2ULL) != 0ULL) {
                break;
            }
            const auto pair_index = normal_index / 2ULL;
            const auto spot_pair = standard_normal_pair(row.seed, 0, pair_index);
            const auto variance_pair =
                standard_normal_pair(row.seed, 1, pair_index);
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                spot_pair.first,
                variance_pair.first,
                spot,
                variance
            );
            max_spot = fmax(max_spot, spot);
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                spot_pair.second,
                variance_pair.second,
                spot,
                variance
            );
            max_spot = fmax(max_spot, spot);
            step += 2U;
        }
        for (; step < num_steps; ++step) {
            const auto normal_index = path * num_steps + step;
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                standard_normal(row.seed, 0, normal_index),
                standard_normal(row.seed, 1, normal_index),
                spot,
                variance
            );
            max_spot = fmax(max_spot, spot);
        }
    } else {
        double log_spot = log(row.spot);
        const auto coefficients = make_qe_coefficients(row, dt);
        const auto first_index = path * num_steps;
        rng::NormalSequence variance_normals(row.seed, 0U, first_index);
        rng::UniformSequence variance_uniforms(row.seed, 2U, first_index);
        rng::NormalSequence independent_normals(row.seed, 1U, first_index);
        for (std::size_t step = 0; step < num_steps; ++step) {
            advance_heston_qe_step_precomputed(
                row,
                coefficients,
                variance_normals.next(),
                variance_uniforms.next(),
                independent_normals.next(),
                log_spot,
                variance
            );
            max_spot = fmax(max_spot, exp(log_spot));
        }
    }
    max_spots[path] = max_spot;
}

__global__ void heston_spot_paths_kernel(
    const HestonRow* row_ptr,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths,
    double* variance_paths
) {
    const auto path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (path >= num_paths) {
        return;
    }

    const auto row = *row_ptr;
    const std::size_t step_count = num_steps + 1U;
    const double dt = row.maturity / static_cast<double>(num_steps);
    const double sqrt_dt = sqrt(dt);
    const double drift_scale = row.risk_free_rate - row.dividend_yield;
    const double correlation_scale = sqrt(1.0 - row.rho * row.rho);
    const double kappa_dt = row.kappa * dt;
    const double vol_of_var_sqrt_dt = row.volatility_of_variance * sqrt_dt;

    double spot = row.spot;
    double variance = row.initial_variance;
    spot_paths[path * step_count] = spot;
    if (variance_paths != nullptr) variance_paths[path * step_count] = variance;
    if (row.scheme == kHestonEulerFullTruncation) {
        std::size_t step = 0;
        for (; step + 1U < num_steps;) {
            const auto normal_index = path * num_steps + step;
            if ((normal_index % 2ULL) != 0ULL) {
                break;
            }
            const auto pair_index = normal_index / 2ULL;
            const auto spot_pair = standard_normal_pair(row.seed, 0, pair_index);
            const auto variance_pair =
                standard_normal_pair(row.seed, 1, pair_index);
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                spot_pair.first,
                variance_pair.first,
                spot,
                variance
            );
            spot_paths[path * step_count + step + 1U] = spot;
            if (variance_paths != nullptr) {
                variance_paths[path * step_count + step + 1U] = variance;
            }
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                spot_pair.second,
                variance_pair.second,
                spot,
                variance
            );
            spot_paths[path * step_count + step + 2U] = spot;
            if (variance_paths != nullptr) {
                variance_paths[path * step_count + step + 2U] = variance;
            }
            step += 2U;
        }
        for (; step < num_steps; ++step) {
            const auto normal_index = path * num_steps + step;
            advance_heston_step(
                row,
                dt,
                sqrt_dt,
                drift_scale,
                correlation_scale,
                kappa_dt,
                vol_of_var_sqrt_dt,
                standard_normal(row.seed, 0, normal_index),
                standard_normal(row.seed, 1, normal_index),
                spot,
                variance
            );
            spot_paths[path * step_count + step + 1U] = spot;
            if (variance_paths != nullptr) {
                variance_paths[path * step_count + step + 1U] = variance;
            }
        }
    } else {
        double log_spot = log(row.spot);
        const auto coefficients = make_qe_coefficients(row, dt);
        const auto first_index = path * num_steps;
        rng::NormalSequence variance_normals(row.seed, 0U, first_index);
        rng::UniformSequence variance_uniforms(row.seed, 2U, first_index);
        rng::NormalSequence independent_normals(row.seed, 1U, first_index);
        for (std::size_t step = 0; step < num_steps; ++step) {
            advance_heston_qe_step_precomputed(
                row,
                coefficients,
                variance_normals.next(),
                variance_uniforms.next(),
                independent_normals.next(),
                log_spot,
                variance
            );
            spot_paths[path * step_count + step + 1U] = exp(log_spot);
            if (variance_paths != nullptr) {
                variance_paths[path * step_count + step + 1U] = variance;
            }
        }
    }
}


}  // namespace

void generate_heston_terminal_spots_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_terminal_spots,
    CudaTiming* timing
) {
    HestonRow* device_row = nullptr;
    double* device_terminal_spots = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    const auto output_bytes = num_paths * sizeof(double);

    check_cuda(cudaMalloc(&device_row, sizeof(HestonRow)), "cudaMalloc row");
    check_cuda(
        cudaMalloc(&device_terminal_spots, output_bytes),
        "cudaMalloc terminal spots"
    );
    try {
        check_cuda(
            cudaMemcpy(
                device_row,
                host_row,
                sizeof(HestonRow),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy row"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");

        const auto block_count =
            static_cast<unsigned int>((num_paths + kThreadsPerBlock - 1U)
                                      / kThreadsPerBlock);
        heston_terminal_spots_kernel<<<block_count, kThreadsPerBlock>>>(
            device_row,
            num_paths,
            num_steps,
            device_terminal_spots
        );
        check_cuda(cudaGetLastError(), "heston terminal kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing");
        check_cuda(
            cudaMemcpy(
                host_terminal_spots,
                device_terminal_spots,
                output_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy terminal spots"
        );
        if (timing != nullptr) {
            timing->simulation_ms = elapsed_ms;
            timing->total_ms = elapsed_ms;
        }
    } catch (...) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(device_row);
        cudaFree(device_terminal_spots);
        throw;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_row);
    cudaFree(device_terminal_spots);
}

void generate_heston_max_spots_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_max_spots,
    CudaTiming* timing
) {
    HestonRow* device_row = nullptr;
    double* device_max_spots = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    const auto output_bytes = num_paths * sizeof(double);

    check_cuda(cudaMalloc(&device_row, sizeof(HestonRow)), "cudaMalloc row");
    check_cuda(
        cudaMalloc(&device_max_spots, output_bytes),
        "cudaMalloc max spots"
    );
    try {
        check_cuda(
            cudaMemcpy(
                device_row,
                host_row,
                sizeof(HestonRow),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy row"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");

        const auto block_count =
            static_cast<unsigned int>((num_paths + kThreadsPerBlock - 1U)
                                      / kThreadsPerBlock);
        heston_max_spots_kernel<<<block_count, kThreadsPerBlock>>>(
            device_row,
            num_paths,
            num_steps,
            device_max_spots
        );
        check_cuda(cudaGetLastError(), "heston max-spots kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing");
        check_cuda(
            cudaMemcpy(
                host_max_spots,
                device_max_spots,
                output_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy max spots"
        );
        if (timing != nullptr) {
            timing->simulation_ms = elapsed_ms;
            timing->total_ms = elapsed_ms;
        }
    } catch (...) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(device_row);
        cudaFree(device_max_spots);
        throw;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_row);
    cudaFree(device_max_spots);
}

void generate_heston_spot_paths_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    CudaTiming* timing
) {
    HestonRow* device_row = nullptr;
    double* device_spot_paths = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    const auto output_bytes = num_paths * (num_steps + 1U) * sizeof(double);

    check_cuda(cudaMalloc(&device_row, sizeof(HestonRow)), "cudaMalloc row");
    check_cuda(
        cudaMalloc(&device_spot_paths, output_bytes),
        "cudaMalloc spot paths"
    );
    try {
        check_cuda(
            cudaMemcpy(
                device_row,
                host_row,
                sizeof(HestonRow),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy row"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");

        const auto block_count =
            static_cast<unsigned int>((num_paths + kThreadsPerBlock - 1U)
                                      / kThreadsPerBlock);
        heston_spot_paths_kernel<<<block_count, kThreadsPerBlock>>>(
            device_row,
            num_paths,
            num_steps,
            device_spot_paths,
            nullptr
        );
        check_cuda(cudaGetLastError(), "heston spot-paths kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing");
        check_cuda(
            cudaMemcpy(
                host_spot_paths,
                device_spot_paths,
                output_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy spot paths"
        );
        if (timing != nullptr) {
            timing->simulation_ms = elapsed_ms;
            timing->total_ms = elapsed_ms;
        }
    } catch (...) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(device_row);
        cudaFree(device_spot_paths);
        throw;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_row);
    cudaFree(device_spot_paths);
}

void generate_heston_state_paths_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    double* host_variance_paths,
    CudaTiming* timing
) {
    HestonRow* device_row = nullptr;
    double* device_spot_paths = nullptr;
    double* device_variance_paths = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};
    const auto output_bytes = num_paths * (num_steps + 1U) * sizeof(double);

    check_cuda(cudaMalloc(&device_row, sizeof(HestonRow)), "cudaMalloc row");
    check_cuda(cudaMalloc(&device_spot_paths, output_bytes), "cudaMalloc spot paths");
    check_cuda(
        cudaMalloc(&device_variance_paths, output_bytes),
        "cudaMalloc variance paths"
    );
    try {
        check_cuda(
            cudaMemcpy(device_row, host_row, sizeof(HestonRow), cudaMemcpyHostToDevice),
            "cudaMemcpy row"
        );
        check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
        check_cuda(cudaEventRecord(start), "cudaEventRecord start");
        const auto block_count = static_cast<unsigned int>(
            (num_paths + kThreadsPerBlock - 1U) / kThreadsPerBlock
        );
        heston_spot_paths_kernel<<<block_count, kThreadsPerBlock>>>(
            device_row, num_paths, num_steps,
            device_spot_paths, device_variance_paths
        );
        check_cuda(cudaGetLastError(), "heston state-paths kernel launch");
        check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
        float elapsed_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing");
        check_cuda(
            cudaMemcpy(
                host_spot_paths, device_spot_paths, output_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy spot paths"
        );
        check_cuda(
            cudaMemcpy(
                host_variance_paths, device_variance_paths, output_bytes,
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy variance paths"
        );
        if (timing != nullptr) {
            timing->simulation_ms = elapsed_ms;
            timing->total_ms = elapsed_ms;
        }
    } catch (...) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(device_row);
        cudaFree(device_spot_paths);
        cudaFree(device_variance_paths);
        throw;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_row);
    cudaFree(device_spot_paths);
    cudaFree(device_variance_paths);
}

}  // namespace ai_factory::cuda
