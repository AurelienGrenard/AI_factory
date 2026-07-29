// Multi-block Heston American-put pricing with GPU Longstaff-Schwartz.
#include "heston/american_put.cuh"

#include "common/check_cuda.cuh"
#include "common/least_squares.cu"
#include "common/reductions.cuh"

// Include the dynamics implementation so NVCC can inline each time step.
#include "heston/dynamics.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::heston {
namespace {

using RegressionBasis = least_squares::TwoFactorLaguerreBasis;

constexpr std::size_t kMomentValueCount = 2U;
constexpr double kRidgeRelative = 1.0e-10;
constexpr double kCholeskyDiagonalFloor = 1.0e-14;
constexpr std::size_t kOneGiB = 1ULL << 30U;

// Prepared model, schedule, and memory location for one batch price.
struct PreparedRow {
    HestonQeParameters initial_stub_model;
    HestonQeParameters regular_model;
    philox::PhiloxKey key;
    std::size_t result_index;
    std::size_t state_offset;
    float strike;
    float initial_spot;
    float inverse_strike;
    float inverse_theta;
    float exercise_discount;
    float initial_discount;
    std::uint32_t exercise_count;
    std::uint32_t initial_stub_steps;
    std::uint32_t steps_per_exercise;
};

// One exercise-homogeneous group of result rows that fits in the workspace.
struct BatchPlan {
    std::size_t result_offset;
    std::size_t result_count;
    std::size_t state_value_count;
    std::uint32_t maximum_exercise_count;
};

// Byte offsets partition one raw allocation into typed device arrays.
struct WorkspaceLayout {
    std::size_t prepared_rows;
    std::size_t exercise_counts;
    std::size_t state_offsets;
    std::size_t observed_spots;
    std::size_t observed_variances;
    std::size_t cashflows;
    std::size_t regression_partials;
    std::size_t regression_coefficients;
    std::size_t regression_valid;
    std::size_t moment_partials;
    std::size_t total_bytes;
};

// Round one byte offset up without relying on implementation padding.
std::size_t aligned_offset(std::size_t offset, std::size_t alignment) {
    const std::size_t remainder = offset % alignment;
    if (remainder == 0U) return offset;
    const std::size_t padding = alignment - remainder;
    if (offset > std::numeric_limits<std::size_t>::max() - padding) {
        throw std::overflow_error("American-put workspace offset exceeds size_t.");
    }
    return offset + padding;
}

// Append one typed array to the workspace and return its aligned byte offset.
template <typename Value>
std::size_t append_workspace_array(
    std::size_t& cursor,
    std::size_t value_count,
    const char* overflow_message
) {
    cursor = aligned_offset(cursor, alignof(Value));
    const std::size_t array_offset = cursor;
    const std::size_t bytes = checked_workspace_product(
        value_count, sizeof(Value), overflow_message
    );
    if (cursor > std::numeric_limits<std::size_t>::max() - bytes) {
        throw std::overflow_error(overflow_message);
    }
    cursor += bytes;
    return array_offset;
}

// Describe every typed region required by one proposed batch.
WorkspaceLayout workspace_layout(
    std::size_t batch_size,
    std::size_t state_value_count,
    std::size_t paths_per_price,
    std::size_t blocks_per_price
) {
    const std::size_t path_value_count = checked_workspace_product(
        batch_size,
        paths_per_price,
        "American-put cashflow count exceeds size_t."
    );
    const std::size_t partial_block_count = checked_workspace_product(
        batch_size,
        blocks_per_price,
        "American-put partial block count exceeds size_t."
    );
    const std::size_t regression_partial_count = checked_workspace_product(
        partial_block_count,
        RegressionBasis::kRegressionValueCount,
        "American-put regression partial count exceeds size_t."
    );
    const std::size_t moment_partial_count = checked_workspace_product(
        partial_block_count,
        kMomentValueCount,
        "American-put moment partial count exceeds size_t."
    );
    const std::size_t coefficient_count = checked_workspace_product(
        batch_size,
        RegressionBasis::kSize,
        "American-put coefficient count exceeds size_t."
    );

    std::size_t cursor = 0U;
    WorkspaceLayout layout{};
    layout.prepared_rows = append_workspace_array<PreparedRow>(
        cursor, batch_size, "American-put prepared rows exceed size_t."
    );
    layout.exercise_counts = append_workspace_array<std::uint32_t>(
        cursor, batch_size, "American-put exercise counts exceed size_t."
    );
    layout.state_offsets = append_workspace_array<std::size_t>(
        cursor, batch_size, "American-put state offsets exceed size_t."
    );
    layout.observed_spots = append_workspace_array<float>(
        cursor, state_value_count, "American-put spot states exceed size_t."
    );
    layout.observed_variances = append_workspace_array<float>(
        cursor, state_value_count, "American-put variance states exceed size_t."
    );
    layout.cashflows = append_workspace_array<float>(
        cursor, path_value_count, "American-put cashflows exceed size_t."
    );
    layout.regression_partials = append_workspace_array<double>(
        cursor,
        regression_partial_count,
        "American-put regression partials exceed size_t."
    );
    layout.regression_coefficients = append_workspace_array<double>(
        cursor, coefficient_count, "American-put coefficients exceed size_t."
    );
    layout.regression_valid = append_workspace_array<std::uint32_t>(
        cursor, batch_size, "American-put regression flags exceed size_t."
    );
    layout.moment_partials = append_workspace_array<double>(
        cursor, moment_partial_count, "American-put moment partials exceed size_t."
    );
    layout.total_bytes = cursor;
    return layout;
}

// Return one typed pointer into the aligned raw CUDA workspace.
template <typename Value>
Value* workspace_pointer(
    unsigned char* workspace,
    std::size_t byte_offset
) {
    return reinterpret_cast<Value*>(workspace + byte_offset);
}

// Match the maturity-anchored exercise schedule used by the device.
std::uint32_t exercise_count(const products::AmericanPutInput& product) {
    const float raw_count = product.maturity / product.exercise_interval;
    const float adjusted =
        raw_count - 8.0f * FLT_EPSILON * std::max(raw_count, 1.0f);
    const double count = std::ceil(static_cast<double>(adjusted));
    if (!(count >= 1.0)
        || count > static_cast<double>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error("American-put exercise count exceeds uint32_t.");
    }
    return static_cast<std::uint32_t>(count);
}

// Pack consecutive result rows into batches under the memory budget.
std::vector<BatchPlan> plan_batches(
    const products::AmericanPutInput* host_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    std::size_t workspace_budget,
    std::size_t& maximum_workspace_bytes,
    std::size_t& maximum_prices_per_batch
) {
    std::vector<BatchPlan> batches;
    std::size_t result_offset = 0U;
    maximum_workspace_bytes = 0U;
    maximum_prices_per_batch = 0U;

    while (result_offset < result_count) {
        std::size_t batch_size = 0U;
        std::size_t state_value_count = 0U;
        std::uint32_t maximum_exercise_count = 0U;

        while (result_offset + batch_size < result_count) {
            const std::size_t result_index = result_offset + batch_size;
            const std::size_t product_index = cartesian_product
                ? result_index % product_count
                : result_index;
            const std::uint32_t row_exercise_count =
                exercise_count(host_products[product_index]);
            const std::size_t row_state_values = checked_workspace_product(
                paths_per_price,
                static_cast<std::size_t>(row_exercise_count - 1U),
                "American-put row state count exceeds size_t."
            );
            if (state_value_count
                > std::numeric_limits<std::size_t>::max() - row_state_values) {
                throw std::overflow_error(
                    "American-put batch state count exceeds size_t."
                );
            }
            const std::size_t candidate_state_values =
                state_value_count + row_state_values;
            const WorkspaceLayout candidate = workspace_layout(
                batch_size + 1U,
                candidate_state_values,
                paths_per_price,
                blocks_per_price
            );
            if (candidate.total_bytes > workspace_budget) break;

            ++batch_size;
            state_value_count = candidate_state_values;
            maximum_exercise_count = std::max(
                maximum_exercise_count, row_exercise_count
            );
        }

        if (batch_size == 0U) {
            const std::size_t result_index = result_offset;
            const std::size_t product_index = cartesian_product
                ? result_index % product_count
                : result_index;
            const std::uint32_t row_exercise_count =
                exercise_count(host_products[product_index]);
            const std::size_t row_state_values = checked_workspace_product(
                paths_per_price,
                static_cast<std::size_t>(row_exercise_count - 1U),
                "American-put row state count exceeds size_t."
            );
            const std::size_t required = workspace_layout(
                1U, row_state_values, paths_per_price, blocks_per_price
            ).total_bytes;
            throw std::runtime_error(
                "American-put result row " + std::to_string(result_index)
                + " requires " + std::to_string(required)
                + " workspace bytes, but only "
                + std::to_string(workspace_budget)
                + " bytes are available. Reduce paths per price."
            );
        }

        const WorkspaceLayout layout = workspace_layout(
            batch_size,
            state_value_count,
            paths_per_price,
            blocks_per_price
        );
        batches.push_back({
            result_offset,
            batch_size,
            state_value_count,
            maximum_exercise_count,
        });
        maximum_workspace_bytes = std::max(
            maximum_workspace_bytes, layout.total_bytes
        );
        maximum_prices_per_batch = std::max(
            maximum_prices_per_batch, batch_size
        );
        result_offset += batch_size;
    }
    return batches;
}

// Prepare each batch row once before thousands of path blocks consume it.
__global__ void prepare_rows_kernel(
    const HestonModelParameters* __restrict__ models,
    const products::AmericanPutInput* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t batch_size,
    float target_dt,
    std::uint64_t base_seed,
    std::size_t result_offset,
    const std::uint32_t* __restrict__ exercise_counts,
    const std::size_t* __restrict__ state_offsets,
    PreparedRow* __restrict__ prepared_rows
) {
    const std::size_t batch_price =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (batch_price >= batch_size) return;

    const std::size_t result_index = result_offset + batch_price;
    const std::size_t model_index = cartesian_product
        ? result_index / product_count
        : result_index;
    const std::size_t product_index = cartesian_product
        ? result_index % product_count
        : result_index;
    const HestonModelParameters model = models[model_index];
    const products::AmericanPutInput product = products[product_index];
    const std::uint32_t row_exercise_count =
        exercise_counts[batch_price];
    const float first_exercise_time = fmaf(
        -static_cast<float>(row_exercise_count - 1U),
        product.exercise_interval,
        product.maturity
    );
    const std::uint32_t initial_stub_steps = static_cast<std::uint32_t>(
        fmaxf(1.0f, ceilf(first_exercise_time / target_dt))
    );
    const std::uint32_t steps_per_exercise = static_cast<std::uint32_t>(
        fmaxf(1.0f, ceilf(product.exercise_interval / target_dt))
    );

    prepared_rows[batch_price] = {
        prepare_model(model, first_exercise_time, initial_stub_steps),
        prepare_model(model, product.exercise_interval, steps_per_exercise),
        philox::make_key(base_seed + result_index),
        result_index,
        state_offsets[batch_price],
        product.strike,
        model.spot,
        1.0f / product.strike,
        1.0f / model.theta,
        expf(-model.risk_free_rate * product.exercise_interval),
        expf(-model.risk_free_rate * first_exercise_time),
        row_exercise_count,
        initial_stub_steps,
        steps_per_exercise,
    };
}

// Simulate every path and initialize its undiscounted maturity cashflow.
__global__ void simulate_paths_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    float* __restrict__ observed_spots,
    float* __restrict__ observed_variances,
    float* __restrict__ cashflows
) {
    __shared__ PreparedRow row;
    if (threadIdx.x == 0U) row = prepared_rows[blockIdx.y];
    __syncthreads();

    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    float* const row_spots = observed_spots + row.state_offset;
    float* const row_variances = observed_variances + row.state_offset;
    float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;

    // Each warp writes consecutive paths at every grid-stride iteration.
    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const float terminal_spot = simulate_spot_variance_on_regular_grid(
            row.initial_stub_model,
            row.regular_model,
            row.key,
            path,
            row.initial_stub_steps,
            row.steps_per_exercise,
            row.exercise_count,
            paths_per_price,
            row_spots,
            row_variances
        );
        row_cashflows[path] = fmaxf(row.strike - terminal_spot, 0.0f);
    }
}

// Produce one deterministic normal-equation partial per path block.
__global__ void regression_partials_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const float* __restrict__ observed_spots,
    const float* __restrict__ observed_variances,
    const float* __restrict__ cashflows,
    double* __restrict__ regression_partials
) {
    __shared__ PreparedRow row;
    if (threadIdx.x == 0U) row = prepared_rows[blockIdx.y];
    __syncthreads();
    if (backward_level >= row.exercise_count) return;

    const std::size_t regression_exercise =
        static_cast<std::size_t>(row.exercise_count - 1U - backward_level);
    const std::size_t state_offset =
        row.state_offset + regression_exercise * paths_per_price;
    const float* const row_spots = observed_spots + state_offset;
    const float* const row_variances = observed_variances + state_offset;
    const float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    double values[RegressionBasis::kRegressionValueCount] = {};

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const float spot = row_spots[path];
        const float immediate = fmaxf(row.strike - spot, 0.0f);
        if (immediate <= 0.0f) continue;

        const RegressionBasis basis =
            RegressionBasis::evaluate(
                spot * row.inverse_strike,
                row_variances[path] * row.inverse_theta
            );
        const double target =
            static_cast<double>(row.exercise_discount)
            * static_cast<double>(row_cashflows[path]);
        std::size_t value_index = 0U;

        #pragma unroll
        for (std::size_t basis_row = 0U;
             basis_row < RegressionBasis::kSize;
             ++basis_row) {
            const double row_value =
                static_cast<double>(basis.values[basis_row]);
            #pragma unroll
            for (std::size_t basis_column = basis_row;
                 basis_column < RegressionBasis::kSize;
                 ++basis_column) {
                values[value_index++] +=
                    row_value
                    * static_cast<double>(basis.values[basis_column]);
            }
            values[RegressionBasis::kGramValueCount + basis_row] +=
                row_value * target;
        }
        values[RegressionBasis::kRegressionValueCount - 1U] += 1.0;
    }

    const double* const totals = reductions::reduce_block_values(values);
    if (threadIdx.x == 0U) {
        const std::size_t batch_price = blockIdx.y;
        #pragma unroll
        for (std::size_t statistic = 0U;
             statistic < RegressionBasis::kRegressionValueCount;
             ++statistic) {
            regression_partials[
                (
                    batch_price * RegressionBasis::kRegressionValueCount
                    + statistic
                )
                    * blocks_per_price
                + blockIdx.x
            ] = totals[statistic];
        }
    }
}

// Reduce all path blocks, regularize G, and solve one system per price.
__global__ void solve_regressions_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t blocks_per_price,
    const double* __restrict__ regression_partials,
    double* __restrict__ regression_coefficients,
    std::uint32_t* __restrict__ regression_valid
) {
    const std::size_t batch_price = blockIdx.x;
    const std::uint32_t exercise_count =
        prepared_rows[batch_price].exercise_count;
    if (backward_level >= exercise_count) return;

    double values[RegressionBasis::kRegressionValueCount] = {};
    #pragma unroll
    for (std::size_t statistic = 0U;
         statistic < RegressionBasis::kRegressionValueCount;
         ++statistic) {
        const double* const partials =
            regression_partials
            + (
                batch_price * RegressionBasis::kRegressionValueCount
                + statistic
            )
                  * blocks_per_price;
        for (std::size_t partial = threadIdx.x;
             partial < blocks_per_price;
             partial += blockDim.x) {
            values[statistic] += partials[partial];
        }
    }

    const double* const totals = reductions::reduce_block_values(values);
    if (threadIdx.x != 0U) return;

    double gram[RegressionBasis::kSize * RegressionBasis::kSize] = {};
    double rhs[RegressionBasis::kSize] = {};
    double* const coefficients =
        regression_coefficients + batch_price * RegressionBasis::kSize;
    std::size_t value_index = 0U;
    double trace = 0.0;

    for (std::size_t row_index = 0U;
         row_index < RegressionBasis::kSize;
         ++row_index) {
        for (std::size_t column = row_index;
             column < RegressionBasis::kSize;
             ++column) {
            const double value = totals[value_index++];
            gram[row_index * RegressionBasis::kSize + column] = value;
            gram[column * RegressionBasis::kSize + row_index] = value;
            if (row_index == column) trace += value;
        }
        rhs[row_index] =
            totals[RegressionBasis::kGramValueCount + row_index];
    }

    const double ridge =
        kRidgeRelative * trace
        / static_cast<double>(RegressionBasis::kSize);
    for (std::size_t index = 0U;
         index < RegressionBasis::kSize;
         ++index) {
        gram[index * RegressionBasis::kSize + index] += ridge;
    }
    const double itm_count =
        totals[RegressionBasis::kRegressionValueCount - 1U];
    const bool solved =
        itm_count > static_cast<double>(RegressionBasis::kSize)
        && least_squares::cholesky_solve_normal_equations(
            gram,
            rhs,
            coefficients,
            RegressionBasis::kSize,
            kCholeskyDiagonalFloor
        );
    regression_valid[batch_price] = solved ? 1U : 0U;
    if (!solved) {
        for (std::size_t index = 0U;
             index < RegressionBasis::kSize;
             ++index) {
            coefficients[index] = 0.0;
        }
    }
}

// Apply one fitted continuation policy and update every discounted cashflow.
__global__ void update_cashflows_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::uint32_t backward_level,
    std::size_t paths_per_price,
    const float* __restrict__ observed_spots,
    const float* __restrict__ observed_variances,
    const double* __restrict__ regression_coefficients,
    const std::uint32_t* __restrict__ regression_valid,
    float* __restrict__ cashflows
) {
    __shared__ PreparedRow row;
    __shared__ float coefficients[RegressionBasis::kSize];
    __shared__ std::uint32_t solved;
    if (threadIdx.x == 0U) {
        row = prepared_rows[blockIdx.y];
        solved = regression_valid[blockIdx.y];
        for (std::size_t index = 0U;
             index < RegressionBasis::kSize;
             ++index) {
            coefficients[index] = static_cast<float>(
                regression_coefficients[
                    blockIdx.y * RegressionBasis::kSize + index
                ]
            );
        }
    }
    __syncthreads();
    if (backward_level >= row.exercise_count) return;

    const std::size_t regression_exercise =
        static_cast<std::size_t>(row.exercise_count - 1U - backward_level);
    const std::size_t state_offset =
        row.state_offset + regression_exercise * paths_per_price;
    const float* const row_spots = observed_spots + state_offset;
    const float* const row_variances = observed_variances + state_offset;
    float* const row_cashflows =
        cashflows + static_cast<std::size_t>(blockIdx.y) * paths_per_price;
    const std::size_t first_path =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t path_stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t path = first_path;
         path < paths_per_price;
         path += path_stride) {
        const float spot = row_spots[path];
        const float immediate = fmaxf(row.strike - spot, 0.0f);
        float updated = row.exercise_discount * row_cashflows[path];

        if (solved != 0U && immediate > 0.0f) {
            const RegressionBasis basis =
                RegressionBasis::evaluate(
                    spot * row.inverse_strike,
                    row_variances[path] * row.inverse_theta
                );
            float continuation = 0.0f;
            #pragma unroll
            for (std::size_t index = 0U;
                 index < RegressionBasis::kSize;
                 ++index) {
                continuation = fmaf(
                    coefficients[index],
                    basis.values[index],
                    continuation
                );
            }
            if (immediate > continuation) {
                updated = immediate;
            }
        }
        row_cashflows[path] = updated;
    }
}

// Produce one FP64 payoff-moment partial per path block.
__global__ void moment_partials_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const float* __restrict__ cashflows,
    double* __restrict__ moment_partials
) {
    const float initial_discount =
        prepared_rows[blockIdx.y].initial_discount;
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
        const float value = initial_discount * row_cashflows[path];
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

// Finish moments and compare continuation with deterministic time-zero exercise.
__global__ void finalize_prices_kernel(
    const PreparedRow* __restrict__ prepared_rows,
    std::size_t paths_per_price,
    std::size_t blocks_per_price,
    const double* __restrict__ moment_partials,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    const std::size_t batch_price = blockIdx.x;
    const double* const row_sums =
        moment_partials
        + (batch_price * kMomentValueCount) * blocks_per_price;
    const double* const row_sumsq =
        moment_partials
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
    const PreparedRow* const row = prepared_rows + batch_price;
    const double immediate = static_cast<double>(
        fmaxf(row->strike - row->initial_spot, 0.0f)
    );
    if (immediate > continuation) {
        prices[row->result_index] = static_cast<float>(immediate);
        standard_errors[row->result_index] = 0.0f;
    } else {
        prices[row->result_index] = static_cast<float>(continuation);
        standard_errors[row->result_index] =
            static_cast<float>(standard_error);
    }
}

// Validate pointers, construction, Monte Carlo dimensions, and the 2D grid.
void validate_launch(
    const HestonModelParameters* device_models,
    std::size_t model_count,
    const products::AmericanPutInput* host_products,
    const products::AmericanPutInput* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(device_standard_errors, "device_standard_errors");
    if (host_products == nullptr) {
        throw std::invalid_argument("host_products is null.");
    }
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    validate_monte_carlo_parameters(paths_per_price, target_dt);
    validate_warp_aligned_block_size(threads_per_block);
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

}  // namespace

// Price every row with one persistent workspace and memory-aware batches.
AmericanPutExecution launch_heston_american_put_cuda(
    const HestonModelParameters* device_models,
    std::size_t model_count,
    const products::AmericanPutInput* host_products,
    const products::AmericanPutInput* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    validate_launch(
        device_models,
        model_count,
        host_products,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        monte_carlo_paths_per_price,
        target_dt,
        threads_per_block,
        blocks_per_price,
        base_seed,
        device_prices,
        device_standard_errors
    );
    const std::size_t path_block_capacity =
        1U + (monte_carlo_paths_per_price - 1U) / threads_per_block;
    const std::size_t launched_blocks_per_price = std::min(
        blocks_per_price, path_block_capacity
    );

    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    check_cuda(
        cudaMemGetInfo(&free_bytes, &total_bytes),
        "cudaMemGetInfo American-put workspace"
    );
    const std::size_t safety_margin = std::max(kOneGiB, free_bytes / 10U);
    if (free_bytes <= safety_margin) {
        throw std::runtime_error(
            "Insufficient free GPU memory after the safety margin."
        );
    }
    const std::size_t workspace_budget = free_bytes - safety_margin;
    std::size_t maximum_workspace_bytes = 0U;
    std::size_t maximum_prices_per_batch = 0U;
    const std::vector<BatchPlan> batches = plan_batches(
        host_products,
        product_count,
        cartesian_product,
        result_count,
        monte_carlo_paths_per_price,
        launched_blocks_per_price,
        workspace_budget,
        maximum_workspace_bytes,
        maximum_prices_per_batch
    );
    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties"
    );
    if (maximum_prices_per_batch
        > static_cast<std::size_t>(properties.maxGridSize[1])) {
        throw std::overflow_error(
            "American-put batch exceeds the current gridDim.y limit."
        );
    }

    unsigned char* device_workspace = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    double kernel_seconds = 0.0;
    std::size_t kernel_launch_count = 0U;
    std::vector<std::uint32_t> host_exercise_counts(
        maximum_prices_per_batch
    );
    std::vector<std::size_t> host_state_offsets(
        maximum_prices_per_batch
    );

    try {
        check_cuda(
            cudaMalloc(&device_workspace, maximum_workspace_bytes),
            "cudaMalloc American-put workspace"
        );
        check_cuda(cudaEventCreate(&start_event), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate stop");

        for (const BatchPlan& batch : batches) {
            std::size_t state_value_cursor = 0U;
            for (std::size_t batch_price = 0U;
                 batch_price < batch.result_count;
                 ++batch_price) {
                const std::size_t result_index =
                    batch.result_offset + batch_price;
                const std::size_t product_index = cartesian_product
                    ? result_index % product_count
                    : result_index;
                const std::uint32_t count = exercise_count(
                    host_products[product_index]
                );
                host_exercise_counts[batch_price] = count;
                host_state_offsets[batch_price] = state_value_cursor;
                state_value_cursor += checked_workspace_product(
                    monte_carlo_paths_per_price,
                    static_cast<std::size_t>(count - 1U),
                    "American-put batch state count exceeds size_t."
                );
            }
            if (state_value_cursor != batch.state_value_count) {
                throw std::logic_error(
                    "American-put batch state count changed after planning."
                );
            }

            const WorkspaceLayout layout = workspace_layout(
                batch.result_count,
                batch.state_value_count,
                monte_carlo_paths_per_price,
                launched_blocks_per_price
            );
            PreparedRow* const prepared_rows =
                workspace_pointer<PreparedRow>(
                    device_workspace, layout.prepared_rows
                );
            std::uint32_t* const device_exercise_counts =
                workspace_pointer<std::uint32_t>(
                    device_workspace, layout.exercise_counts
                );
            std::size_t* const device_state_offsets =
                workspace_pointer<std::size_t>(
                    device_workspace, layout.state_offsets
                );
            float* const observed_spots = workspace_pointer<float>(
                device_workspace, layout.observed_spots
            );
            float* const observed_variances = workspace_pointer<float>(
                device_workspace, layout.observed_variances
            );
            float* const cashflows = workspace_pointer<float>(
                device_workspace, layout.cashflows
            );
            double* const regression_partials = workspace_pointer<double>(
                device_workspace, layout.regression_partials
            );
            double* const regression_coefficients = workspace_pointer<double>(
                device_workspace, layout.regression_coefficients
            );
            std::uint32_t* const regression_valid =
                workspace_pointer<std::uint32_t>(
                    device_workspace, layout.regression_valid
                );
            double* const moment_partials = workspace_pointer<double>(
                device_workspace, layout.moment_partials
            );

            check_cuda(
                cudaMemcpy(
                    device_exercise_counts,
                    host_exercise_counts.data(),
                    batch.result_count * sizeof(std::uint32_t),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy American-put exercise counts"
            );
            check_cuda(
                cudaMemcpy(
                    device_state_offsets,
                    host_state_offsets.data(),
                    batch.result_count * sizeof(std::size_t),
                    cudaMemcpyHostToDevice
                ),
                "cudaMemcpy American-put state offsets"
            );

            const dim3 path_grid(
                static_cast<unsigned int>(launched_blocks_per_price),
                static_cast<unsigned int>(batch.result_count),
                1U
            );
            const unsigned int row_blocks = static_cast<unsigned int>(
                (batch.result_count + threads_per_block - 1U)
                / threads_per_block
            );
            const std::size_t warp_count = threads_per_block / 32U;
            const std::size_t regression_shared_bytes =
                (
                    RegressionBasis::kRegressionValueCount * warp_count
                    + RegressionBasis::kRegressionValueCount
                )
                * sizeof(double);
            const std::size_t moment_shared_bytes =
                2U * warp_count * sizeof(double);

            check_cuda(cudaEventRecord(start_event), "cudaEventRecord start");

            prepare_rows_kernel<<<row_blocks, threads_per_block>>>(
                device_models,
                device_products,
                product_count,
                cartesian_product,
                batch.result_count,
                target_dt,
                base_seed,
                batch.result_offset,
                device_exercise_counts,
                device_state_offsets,
                prepared_rows
            );
            check_cuda(cudaGetLastError(), "prepare American-put rows");
            ++kernel_launch_count;

            simulate_paths_kernel<<<path_grid, threads_per_block>>>(
                prepared_rows,
                monte_carlo_paths_per_price,
                observed_spots,
                observed_variances,
                cashflows
            );
            check_cuda(cudaGetLastError(), "simulate American-put paths");
            ++kernel_launch_count;

            for (std::uint32_t backward_level = 1U;
                 backward_level < batch.maximum_exercise_count;
                 ++backward_level) {
                regression_partials_kernel<<<
                    path_grid,
                    threads_per_block,
                    regression_shared_bytes
                >>>(
                    prepared_rows,
                    backward_level,
                    monte_carlo_paths_per_price,
                    launched_blocks_per_price,
                    observed_spots,
                    observed_variances,
                    cashflows,
                    regression_partials
                );
                check_cuda(
                    cudaGetLastError(), "American-put regression partials"
                );

                solve_regressions_kernel<<<
                    static_cast<unsigned int>(batch.result_count),
                    threads_per_block,
                    regression_shared_bytes
                >>>(
                    prepared_rows,
                    backward_level,
                    launched_blocks_per_price,
                    regression_partials,
                    regression_coefficients,
                    regression_valid
                );
                check_cuda(
                    cudaGetLastError(), "solve American-put regressions"
                );

                update_cashflows_kernel<<<path_grid, threads_per_block>>>(
                    prepared_rows,
                    backward_level,
                    monte_carlo_paths_per_price,
                    observed_spots,
                    observed_variances,
                    regression_coefficients,
                    regression_valid,
                    cashflows
                );
                check_cuda(
                    cudaGetLastError(), "update American-put cashflows"
                );
                kernel_launch_count += 3U;
            }

            moment_partials_kernel<<<
                path_grid,
                threads_per_block,
                moment_shared_bytes
            >>>(
                prepared_rows,
                monte_carlo_paths_per_price,
                launched_blocks_per_price,
                cashflows,
                moment_partials
            );
            check_cuda(cudaGetLastError(), "American-put moment partials");
            ++kernel_launch_count;

            finalize_prices_kernel<<<
                static_cast<unsigned int>(batch.result_count),
                threads_per_block,
                moment_shared_bytes
            >>>(
                prepared_rows,
                monte_carlo_paths_per_price,
                launched_blocks_per_price,
                moment_partials,
                device_prices,
                device_standard_errors
            );
            check_cuda(cudaGetLastError(), "finalize American-put prices");
            ++kernel_launch_count;

            check_cuda(cudaEventRecord(stop_event), "cudaEventRecord stop");
            check_cuda(
                cudaEventSynchronize(stop_event),
                "cudaEventSynchronize stop"
            );
            float batch_milliseconds = 0.0f;
            check_cuda(
                cudaEventElapsedTime(
                    &batch_milliseconds, start_event, stop_event
                ),
                "cudaEventElapsedTime American-put batch"
            );
            kernel_seconds +=
                static_cast<double>(batch_milliseconds) * 1.0e-3;
        }

        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy start");
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy stop");
        check_cuda(
            cudaFree(device_workspace), "cudaFree American-put workspace"
        );
    } catch (...) {
        if (start_event != nullptr) cudaEventDestroy(start_event);
        if (stop_event != nullptr) cudaEventDestroy(stop_event);
        if (device_workspace != nullptr) cudaFree(device_workspace);
        throw;
    }

    return {
        kernel_seconds,
        batches.size(),
        kernel_launch_count,
        maximum_prices_per_batch,
        launched_blocks_per_price,
        maximum_workspace_bytes,
    };
}

}  // namespace ai_factory::workbench::heston
