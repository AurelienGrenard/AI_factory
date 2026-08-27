// Shared multi-block CUDA engine for Longstaff-Schwartz pricing policies.
#pragma once

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/longstaff_schwartz/concepts.cuh"
#include "common/longstaff_schwartz/exercise_decision.cuh"
#include "common/longstaff_schwartz/execution_plan.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/reductions.cuh"
#include "common/simulation/schedule.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::longstaff_schwartz {

inline constexpr std::size_t kMaximumSharedPreparedRowBytes = 2048U;
inline constexpr std::size_t kMomentValueCount = 2U;

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
__global__ void prepare_rows_kernel(
    typename PricingPolicy::DeviceInputs inputs,
    std::size_t result_offset,
    std::size_t batch_size,
    typename PricingPolicy::Schedule::TimeConfiguration time_configuration,
    std::uint64_t base_seed,
    std::size_t paths_per_price,
    const std::size_t* __restrict__ state_offsets,
    typename PricingPolicy::PreparedRow* __restrict__ prepared_rows
) {
    const std::size_t batch_price =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch_price >= batch_size) return;

    const std::size_t result_index = result_offset + batch_price;
    prepared_rows[batch_price] =
        inputs.template prepare_row<PricingPolicy>(
            result_index,
            time_configuration,
            philox::make_key(base_seed + result_index),
            result_index,
            state_offsets[batch_price],
            paths_per_price
        );
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
__global__ void simulate_paths_kernel(
    const typename PricingPolicy::PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    typename PricingPolicy::StateView states,
    float* __restrict__ cashflows
) {
    static_assert(
        sizeof(typename PricingPolicy::PreparedRow)
            <= kMaximumSharedPreparedRowBytes,
        "Longstaff-Schwartz PreparedRow exceeds the 2048-byte shared-memory "
        "budget; store a compact schedule/state view over device-resident "
        "pools."
    );
    __shared__ typename PricingPolicy::PreparedRow row;
    if (threadIdx.x == 0U) row = prepared_rows[blockIdx.y];
    __syncthreads();

    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        row_cashflows[path] = PricingPolicy::simulate_path(
            row,
            path,
            paths_per_price,
            states
        );
    }
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
__global__ void regression_partials_kernel(
    const typename PricingPolicy::PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    typename PricingPolicy::StateView states,
    const float* __restrict__ cashflows,
    double* __restrict__ regression_partials
) {
    __shared__ typename PricingPolicy::PreparedRow row;
    if (threadIdx.x == 0U) row = prepared_rows[blockIdx.y];
    __syncthreads();
    if (backward_level >= row.regression_count) return;

    const float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    double statistics[Regressor::kRegressionValueCount] = {};
    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const std::size_t observation = PricingPolicy::state_index(
            row,
            backward_level,
            paths_per_price,
            path
        );
        const float immediate = PricingPolicy::immediate_value(
            row, states, observation
        );
        if (!PricingPolicy::regression_candidate(immediate)) continue;
        const typename Regressor::Features features = Regressor::evaluate(
            PricingPolicy::regression_input(row, states, observation)
        );
        Regressor::accumulate(
            features,
            PricingPolicy::regression_target(
                row,
                states,
                observation,
                row_cashflows[path]
            ),
            statistics
        );
    }
    Regressor::reduce_and_store_partials(
        statistics,
        blockIdx.y,
        blockIdx.x,
        blocks_per_price,
        regression_partials
    );
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
__global__ void solve_regressions_kernel(
    const typename PricingPolicy::PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t blocks_per_price,
    const double* __restrict__ regression_partials,
    double* __restrict__ regression_coefficients,
    RegressionStatus* __restrict__ regression_statuses,
    RegressionDiagnostics* __restrict__ regression_diagnostics
) {
    const std::size_t batch_price = blockIdx.x;
    Regressor::solve_for_row(
        prepared_rows[batch_price].regression_count,
        backward_level,
        batch_price,
        blocks_per_price,
        regression_partials,
        regression_coefficients,
        regression_statuses,
        regression_diagnostics
    );
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
__global__ void update_cashflows_kernel(
    const typename PricingPolicy::PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t paths_per_price,
    typename PricingPolicy::StateView states,
    const double* __restrict__ regression_coefficients,
    const RegressionStatus* __restrict__ regression_statuses,
    float* __restrict__ cashflows
) {
    __shared__ typename PricingPolicy::PreparedRow row;
    __shared__ double coefficients[Regressor::kBasisSize];
    __shared__ RegressionStatus regression_status;
    if (threadIdx.x == 0U) {
        row = prepared_rows[blockIdx.y];
        regression_status = regression_statuses[blockIdx.y];
        for (std::size_t index = 0U;
             index < Regressor::kBasisSize;
             ++index) {
            coefficients[index] = regression_coefficients[
                blockIdx.y * Regressor::kBasisSize + index
            ];
        }
    }
    __syncthreads();
    if (backward_level >= row.regression_count) return;

    float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const std::size_t observation = PricingPolicy::state_index(
            row,
            backward_level,
            paths_per_price,
            path
        );
        const float immediate = PricingPolicy::immediate_value(
            row, states, observation
        );
        float updated = PricingPolicy::continued_cashflow(
            row,
            states,
            observation,
            row_cashflows[path]
        );

        if (regression_status == RegressionStatus::success
            && PricingPolicy::regression_candidate(immediate)) {
            const typename Regressor::Features features =
                Regressor::evaluate(
                    PricingPolicy::regression_input(
                        row, states, observation
                    )
                );
            const double continuation = Regressor::predict(
                features, coefficients
            );
            updated = select_exercise_cashflow(
                immediate, continuation, updated
            );
        }
        row_cashflows[path] = updated;
    }
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
__global__ void moment_partials_kernel(
    const typename PricingPolicy::PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    typename PricingPolicy::StateView states,
    const float* __restrict__ cashflows,
    double* __restrict__ moment_partials
) {
    const typename PricingPolicy::PreparedRow& row =
        prepared_rows[blockIdx.y];
    const float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    double sum = 0.0;
    double sumsq = 0.0;

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const float value = PricingPolicy::initial_continuation_value(
            row,
            states,
            path,
            paths_per_price,
            row_cashflows[path]
        );
        const double accumulated_value = static_cast<double>(value);
        sum += accumulated_value;
        sumsq += accumulated_value * accumulated_value;
    }

    const reductions::MomentSums totals =
        reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0U) {
        const std::size_t batch_price = blockIdx.y;
        moment_partials[
            (batch_price * kMomentValueCount) * blocks_per_price + blockIdx.x
        ] = totals.sum;
        moment_partials[
            (batch_price * kMomentValueCount + 1U) * blocks_per_price
                + blockIdx.x
        ] = totals.sumsq;
    }
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
__global__ void finalize_prices_kernel(
    const typename PricingPolicy::PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const double* __restrict__ moment_partials,
    const RegressionDiagnostics* __restrict__ regression_diagnostics,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    const std::size_t batch_price = blockIdx.x;
    if (regression_diagnostics[batch_price].fatal_failure_count != 0U) {
        if (threadIdx.x == 0U) {
            const std::size_t result_index =
                prepared_rows[batch_price].result_index;
            invalidate_regression_result_if_fatal(
                regression_diagnostics[batch_price],
                result_index,
                prices,
                standard_errors
            );
        }
        return;
    }
    const double* const row_sums = moment_partials
        + (batch_price * kMomentValueCount) * blocks_per_price;
    const double* const row_sumsq = moment_partials
        + (batch_price * kMomentValueCount + 1U) * blocks_per_price;
    double sum = 0.0;
    double sumsq = 0.0;

    for (std::size_t partial = threadIdx.x;
         partial < blocks_per_price;
         partial += blockDim.x) {
        sum += row_sums[partial];
        sumsq += row_sumsq[partial];
    }
    const reductions::MomentSums totals =
        reductions::reduce_block(sum, sumsq);
    if (threadIdx.x != 0U) return;

    double continuation = 0.0;
    double standard_error = 0.0;
    reductions::compute_statistics(
        totals, paths_per_price, continuation, standard_error
    );
    const typename PricingPolicy::PreparedRow& row =
        prepared_rows[batch_price];
    const double immediate = static_cast<double>(
        PricingPolicy::initial_exercise_value(row)
    );
    if (immediate > continuation) {
        prices[row.result_index] = static_cast<float>(immediate);
        standard_errors[row.result_index] = 0.0f;
    } else {
        prices[row.result_index] = static_cast<float>(continuation);
        standard_errors[row.result_index] =
            static_cast<float>(standard_error);
    }
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
inline void validate_longstaff_schwartz_launch(
    const typename PricingPolicy::DeviceInputs& device_inputs,
    const typename PricingPolicy::HostInputs& host_inputs,
    std::size_t result_count,
    std::size_t paths_per_price,
    const typename PricingPolicy::Schedule::TimeConfiguration&
        time_configuration,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    static_assert(
        sizeof(typename PricingPolicy::PreparedRow)
            <= kMaximumSharedPreparedRowBytes,
        "Longstaff-Schwartz PreparedRow exceeds the 2048-byte shared-memory "
        "budget; store a compact schedule/state view over device-resident "
        "pools."
    );
    device_inputs.validate(result_count);
    host_inputs.validate(result_count);
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(
        device_standard_errors, "device_standard_errors"
    );
    validate_monte_carlo_path_count(paths_per_price);
    simulation::validate_time_configuration(time_configuration);
    validate_reduction_block_size(threads_per_block);
    validate_row_seed_range(result_count, base_seed);

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );
    if (blocks_per_price == 0U
        || blocks_per_price
            > static_cast<std::size_t>(properties.maxGridSize[0])) {
        throw std::invalid_argument(
            "blocks_per_price must fit the current gridDim.x limit."
        );
    }
}

template<
    EarlyExercisePricingPolicy PricingPolicy,
    SmallLinearRegressor Regressor
>
requires LongstaffSchwartzPolicy<PricingPolicy, Regressor>
LaunchResult launch_longstaff_schwartz_cuda(
    const typename PricingPolicy::DeviceInputs& device_inputs,
    const typename PricingPolicy::HostInputs& host_inputs,
    std::size_t result_count,
    std::size_t paths_per_price,
    const typename PricingPolicy::Schedule::TimeConfiguration&
        time_configuration,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* product_name
) {
    validate_longstaff_schwartz_launch<PricingPolicy, Regressor>(
        device_inputs,
        host_inputs,
        result_count,
        paths_per_price,
        time_configuration,
        threads_per_block,
        blocks_per_price,
        base_seed,
        device_prices,
        device_standard_errors
    );

    const std::size_t path_block_capacity =
        1U + (paths_per_price - 1U) / threads_per_block;
    const std::size_t launched_blocks_per_price = std::min(
        blocks_per_price, path_block_capacity
    );
    const WorkspaceBudget budget = query_workspace_budget(product_name);
    const ExecutionPlan plan =
        make_execution_plan<PricingPolicy, Regressor>(
            host_inputs,
            result_count,
            paths_per_price,
            launched_blocks_per_price,
            budget.available_bytes,
            product_name
        );

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );
    if (plan.maximum_prices_per_batch
        > static_cast<std::size_t>(properties.maxGridSize[1])) {
        throw std::overflow_error(
            std::string(product_name)
            + " batch exceeds the current gridDim.y limit."
        );
    }

    LaunchResources resources(plan.maximum_workspace_bytes, product_name);
    double kernel_seconds = 0.0;
    std::size_t kernel_launch_count = 0U;
    std::vector<std::size_t> host_state_offsets(
        plan.maximum_prices_per_batch
    );
    std::vector<RegressionDiagnostics> host_regression_diagnostics(
        plan.maximum_prices_per_batch
    );
    RegressionDiagnosticSummary regression_summary{};

    const std::string prepare_name =
        std::string(diagnostic_name) + ".prepare_rows";
    const std::string simulation_name =
        std::string(diagnostic_name) + ".simulate_paths";
    const std::string partials_name =
        std::string(diagnostic_name) + ".regression_partials";
    const std::string solve_name =
        std::string(diagnostic_name) + ".solve_regressions";
    const std::string update_name =
        std::string(diagnostic_name) + ".update_cashflows";
    const std::string moments_name =
        std::string(diagnostic_name) + ".moment_partials";
    const std::string finalize_name =
        std::string(diagnostic_name) + ".finalize_prices";

    for (const BatchPlan& batch : plan.batches) {
        std::size_t state_value_cursor = 0U;
        for (std::size_t batch_price = 0U;
             batch_price < batch.result_count;
             ++batch_price) {
            const EarlyExerciseRowPlan& row =
                plan.rows[batch.result_offset + batch_price];
            host_state_offsets[batch_price] = state_value_cursor;
            state_value_cursor += row.state_value_count;
        }
        if (state_value_cursor != batch.state_value_count) {
            throw std::logic_error(
                std::string(product_name)
                + " batch state count changed after planning."
            );
        }

        const WorkspaceLayout layout = make_workspace_layout(
            plan.descriptor,
            batch.result_count,
            batch.state_value_count,
            paths_per_price,
            launched_blocks_per_price,
            product_name
        );
        unsigned char* const device_workspace = resources.workspace();
        typename PricingPolicy::PreparedRow* const prepared_rows =
            workspace_pointer<typename PricingPolicy::PreparedRow>(
                device_workspace, layout.prepared_rows
            );
        std::size_t* const device_state_offsets =
            workspace_pointer<std::size_t>(
                device_workspace, layout.state_offsets
            );
        const typename PricingPolicy::StateView states =
            PricingPolicy::make_state_view(device_workspace, layout);
        float* const cashflows = workspace_pointer<float>(
            device_workspace, layout.cashflows
        );
        double* const regression_partials = workspace_pointer<double>(
            device_workspace, layout.regression_partials
        );
        double* const regression_coefficients = workspace_pointer<double>(
            device_workspace, layout.regression_coefficients
        );
        RegressionStatus* const regression_statuses =
            workspace_pointer<RegressionStatus>(
                device_workspace, layout.regression_statuses
            );
        RegressionDiagnostics* const regression_diagnostics =
            workspace_pointer<RegressionDiagnostics>(
                device_workspace, layout.regression_diagnostics
            );
        double* const moment_partials = workspace_pointer<double>(
            device_workspace, layout.moment_partials
        );

        check_cuda(
            cudaMemcpy(
                device_state_offsets,
                host_state_offsets.data(),
                batch.result_count * sizeof(std::size_t),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy early-exercise state offsets"
        );
        check_cuda(
            cudaMemset(
                regression_statuses,
                0,
                batch.result_count * sizeof(RegressionStatus)
            ),
            "cudaMemset early-exercise regression statuses"
        );
        check_cuda(
            cudaMemset(
                regression_diagnostics,
                0,
                batch.result_count * sizeof(RegressionDiagnostics)
            ),
            "cudaMemset early-exercise regression diagnostics"
        );

        const dim3 path_grid(
            static_cast<unsigned int>(launched_blocks_per_price),
            static_cast<unsigned int>(batch.result_count)
        );
        const unsigned int row_blocks = static_cast<unsigned int>(
            1U + (batch.result_count - 1U) / threads_per_block
        );
        const std::size_t regression_shared_bytes =
            Regressor::shared_bytes(threads_per_block);
        const std::size_t moment_shared_bytes =
            2U * (threads_per_block / 32U) * sizeof(double);

        resources.start_batch();

        report_cuda_kernel_launch_if_enabled(
            prepare_name.c_str(),
            "common",
            prepare_rows_kernel<PricingPolicy, Regressor>,
            dim3(row_blocks),
            dim3(threads_per_block),
            0U
        );
        prepare_rows_kernel<PricingPolicy, Regressor><<<
            row_blocks, threads_per_block
        >>>(
            device_inputs,
            batch.result_offset,
            batch.result_count,
            time_configuration,
            base_seed,
            paths_per_price,
            device_state_offsets,
            prepared_rows
        );
        check_cuda(cudaGetLastError(), "prepare early-exercise rows");
        ++kernel_launch_count;

        report_cuda_kernel_launch_if_enabled(
            simulation_name.c_str(),
            diagnostic_variant,
            simulate_paths_kernel<PricingPolicy, Regressor>,
            path_grid,
            dim3(threads_per_block),
            0U
        );
        simulate_paths_kernel<PricingPolicy, Regressor><<<
            path_grid, threads_per_block
        >>>(
            prepared_rows,
            paths_per_price,
            states,
            cashflows
        );
        check_cuda(cudaGetLastError(), "simulate early-exercise paths");
        ++kernel_launch_count;

        for (std::uint32_t backward_level = 0U;
             backward_level < batch.maximum_regression_count;
             ++backward_level) {
            report_cuda_kernel_launch_if_enabled(
                partials_name.c_str(),
                diagnostic_variant,
                regression_partials_kernel<PricingPolicy, Regressor>,
                path_grid,
                dim3(threads_per_block),
                regression_shared_bytes
            );
            regression_partials_kernel<PricingPolicy, Regressor><<<
                path_grid,
                threads_per_block,
                regression_shared_bytes
            >>>(
                prepared_rows,
                backward_level,
                paths_per_price,
                launched_blocks_per_price,
                states,
                cashflows,
                regression_partials
            );
            check_cuda(
                cudaGetLastError(),
                "accumulate early-exercise regression"
            );

            report_cuda_kernel_launch_if_enabled(
                solve_name.c_str(),
                "common",
                solve_regressions_kernel<PricingPolicy, Regressor>,
                dim3(static_cast<unsigned int>(batch.result_count)),
                dim3(threads_per_block),
                regression_shared_bytes
            );
            solve_regressions_kernel<PricingPolicy, Regressor><<<
                static_cast<unsigned int>(batch.result_count),
                threads_per_block,
                regression_shared_bytes
            >>>(
                prepared_rows,
                backward_level,
                launched_blocks_per_price,
                regression_partials,
                regression_coefficients,
                regression_statuses,
                regression_diagnostics
            );
            check_cuda(cudaGetLastError(), "solve early-exercise regression");

            report_cuda_kernel_launch_if_enabled(
                update_name.c_str(),
                diagnostic_variant,
                update_cashflows_kernel<PricingPolicy, Regressor>,
                path_grid,
                dim3(threads_per_block),
                0U
            );
            update_cashflows_kernel<PricingPolicy, Regressor><<<
                path_grid, threads_per_block
            >>>(
                prepared_rows,
                backward_level,
                paths_per_price,
                states,
                regression_coefficients,
                regression_statuses,
                cashflows
            );
            check_cuda(cudaGetLastError(), "update early-exercise cashflows");
            kernel_launch_count += 3U;
        }

        report_cuda_kernel_launch_if_enabled(
            moments_name.c_str(),
            "common",
            moment_partials_kernel<PricingPolicy, Regressor>,
            path_grid,
            dim3(threads_per_block),
            moment_shared_bytes
        );
        moment_partials_kernel<PricingPolicy, Regressor><<<
            path_grid, threads_per_block, moment_shared_bytes
        >>>(
            prepared_rows,
            paths_per_price,
            launched_blocks_per_price,
            states,
            cashflows,
            moment_partials
        );
        check_cuda(cudaGetLastError(), "reduce early-exercise moments");
        ++kernel_launch_count;

        report_cuda_kernel_launch_if_enabled(
            finalize_name.c_str(),
            diagnostic_variant,
            finalize_prices_kernel<PricingPolicy, Regressor>,
            dim3(static_cast<unsigned int>(batch.result_count)),
            dim3(threads_per_block),
            moment_shared_bytes
        );
        finalize_prices_kernel<PricingPolicy, Regressor><<<
            static_cast<unsigned int>(batch.result_count),
            threads_per_block,
            moment_shared_bytes
        >>>(
            prepared_rows,
            paths_per_price,
            launched_blocks_per_price,
            moment_partials,
            regression_diagnostics,
            device_prices,
            device_standard_errors
        );
        check_cuda(cudaGetLastError(), "finalize early-exercise prices");
        ++kernel_launch_count;

        kernel_seconds += resources.finish_batch();
        check_cuda(
            cudaMemcpy(
                host_regression_diagnostics.data(),
                regression_diagnostics,
                batch.result_count * sizeof(RegressionDiagnostics),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy early-exercise regression diagnostics"
        );
        for (std::size_t batch_price = 0U;
             batch_price < batch.result_count;
             ++batch_price) {
            const RegressionDiagnostics& diagnostics =
                host_regression_diagnostics[batch_price];
            regression_summary.successful_regression_count +=
                diagnostics.successful_regression_count;
            regression_summary.no_candidate_count +=
                diagnostics.no_candidate_count;
            regression_summary.insufficient_candidate_count +=
                diagnostics.insufficient_candidate_count;
            regression_summary.non_finite_statistics_count +=
                diagnostics.non_finite_statistics_count;
            regression_summary.factorization_failure_count +=
                diagnostics.factorization_failure_count;
            regression_summary.non_finite_coefficient_count +=
                diagnostics.non_finite_coefficient_count;
            if (diagnostics.fatal_failure_count == 0U) continue;

            ++regression_summary.affected_result_count;
            if (regression_summary.first_fatal_result_index
                != std::numeric_limits<std::size_t>::max()) {
                continue;
            }
            regression_summary.first_fatal_result_index =
                batch.result_offset + batch_price;
            if (diagnostics.non_finite_statistics_count != 0U) {
                regression_summary.first_fatal_status =
                    RegressionStatus::non_finite_statistics;
            } else if (diagnostics.factorization_failure_count != 0U) {
                regression_summary.first_fatal_status =
                    RegressionStatus::factorization_failure;
            } else {
                regression_summary.first_fatal_status =
                    RegressionStatus::non_finite_coefficients;
            }
        }
    }

    return {
        kernel_seconds,
        plan.batches.size(),
        kernel_launch_count,
        plan.maximum_prices_per_batch,
        launched_blocks_per_price,
        plan.maximum_workspace_bytes,
        regression_summary,
    };
}

}  // namespace ai_factory::workbench::longstaff_schwartz
