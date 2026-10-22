// One or several blocks per price; one or several threads own each path.
#pragma once

#include "cuda/check_cuda.cuh"
#include "monte_carlo/monte_carlo_kernel.cuh"
#include "multi_asset/limits.cuh"
#include "philox/distributions.cuh"
#include "monte_carlo/reductions.cuh"
#include "multi_asset/markovian_pack.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo {

template<typename Pack, typename Product>
struct MultiAssetDeviceInputs {
    typename Pack::Inputs models;
    const typename Product::ProductParameters* products;
    const float* packed_factors;
    const RowIndexSelector* row_selectors;
};

__host__ __device__ constexpr std::size_t align_workspace(
    std::size_t bytes, std::size_t alignment
) {
    return (bytes + alignment - 1U) & ~(alignment - 1U);
}

template<typename Pack, typename Product, unsigned int TeamSize>
__device__ __forceinline__ float evaluate_multi_asset_path(
    typename Pack::States& states,
    const typename Pack::Prepared& prepared,
    const typename Product::PreparedProduct& prepared_product,
    const typename Product::ProductParameters& product_parameters,
    const float* lower,
    float* team_innovations,
    philox::PhiloxKey key,
    std::uint64_t path,
    std::uint32_t steps,
    std::uint32_t steps_per_day,
    unsigned int lane
) {
    constexpr std::size_t dimension = Pack::kFactorCount;
    Pack::template initial_states<0U, TeamSize>(
        states, prepared, lane
    );
    auto handler = Product::make_handler(prepared_product);
    for (std::uint32_t step = 0U; step < steps; ++step) {
        simulation::BrownianNormals<dimension, TeamSize> normals;
        normals.generate(key, path, step, lane);
        simulation::CorrelatedBrownianStep<dimension, TeamSize> brownian;
        if constexpr (TeamSize>1U) {
            if(team_innovations) {
                normals.store_to_shared(team_innovations,lane);
                brownian.calculate_from_shared(
                    team_innovations,lower,lane);
                brownian.publish_to_shared(team_innovations,lane);
            } else brownian.calculate(normals,lower,lane);
        } else brownian.calculate(normals,lower,lane);
        Pack::template advance_one_step<
            0U, 0U, dimension, TeamSize
        >(states, prepared, brownian, key, path, step, lane);
        if (Product::is_observation_step(
                step, steps, steps_per_day, product_parameters
            )) {
            const auto observation =
                Product::template observe<Pack, TeamSize>(
                    states, prepared, lane
                );
            bool keep_going = true;
            if (lane == 0U) {
                keep_going = handler.on_observation(step, observation);
            }
            if constexpr (TeamSize > 1U) {
                keep_going = __shfl_sync(
                    simulation::subgroup_mask<TeamSize>(),
                    keep_going, 0U, TeamSize
                );
            }
            if (!keep_going) break;
        }
    }
    return lane == 0U
        ? Product::finalize(prepared_product, handler) : 0.0f;
}

template<typename Pack, typename Product, unsigned int TeamSize>
__global__ void multi_asset_price_kernel(
    MultiAssetDeviceInputs<Pack, Product> inputs,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
    std::uint64_t base_seed,
    float* prices,
    float* standard_errors,
    reductions::MomentSums* partial_moments,
    bool use_shared_normals
) {
    static_assert(Pack::kAssetCount > 0U);
    static_assert(TeamSize > 0U && TeamSize <= 32U
        && (TeamSize & (TeamSize - 1U)) == 0U);
    constexpr std::size_t lower_count =
        Pack::kFactorCount * (Pack::kFactorCount + 1U) / 2U;
    __shared__ typename Pack::Prepared prepared;
    __shared__ typename Product::PreparedProduct prepared_product;
    __shared__ typename Product::ProductParameters product_parameters;
    __shared__ std::uint32_t product_index;
    __shared__ std::uint32_t correlation_index;
    __shared__ philox::PhiloxKey key;
    __shared__ std::uint32_t steps;

    extern __shared__ __align__(16) unsigned char workspace[];
    const std::size_t warp_count = blockDim.x / 32U;
    const std::size_t lower_offset = align_workspace(
        2U * warp_count * sizeof(double), alignof(float)
    );
    const std::size_t state_offset = align_workspace(
        lower_offset + lower_count * sizeof(float),
        alignof(typename Pack::States)
    );
    auto* lower = reinterpret_cast<float*>(workspace + lower_offset);
    auto* team_states = reinterpret_cast<typename Pack::States*>(
        workspace + state_offset
    );
    const std::size_t normal_offset=align_workspace(
        state_offset+(TeamSize==1U ? 0U : blockDim.x/TeamSize)
            *sizeof(typename Pack::States),alignof(float));
    auto* team_innovations=reinterpret_cast<float*>(
        workspace+normal_offset);
    const unsigned int lane = threadIdx.x % TeamSize;
    const unsigned int team = threadIdx.x / TeamSize;
    const unsigned int teams_per_block = blockDim.x / TeamSize;

    for (std::size_t launch_index = blockIdx.x;
         launch_index < launch_result_count;
         launch_index += gridDim.x) {
        const std::size_t result_index = result_offset + launch_index;
        if (threadIdx.x == 0U) {
            product_index = inputs.row_selectors[Pack::kAssetCount].select(
                static_cast<std::uint32_t>(result_index));
            correlation_index = inputs.row_selectors[
                Pack::kAssetCount + 1U].select(
                    static_cast<std::uint32_t>(result_index));
            product_parameters = inputs.products[product_index];
            prepared_product = Product::prepare(product_parameters, 1.0f);
            steps = product_parameters.maturity_days
                * simulation_steps_per_day;
            key = philox::make_key(base_seed + result_index);
        }
        __syncthreads();
        const float* factor = inputs.packed_factors
            + static_cast<std::size_t>(correlation_index) * lower_count;
        for (std::size_t index = threadIdx.x;
             index < lower_count; index += blockDim.x) {
            lower[index] = factor[index];
        }
        __syncthreads();
        if (threadIdx.x == 0U)
            Pack::prepare_selected(
                prepared, inputs.models,
                static_cast<std::uint32_t>(result_index), dt, lower);
        __syncthreads();

        // A logical bin is path % blockDim.x, regardless of TeamSize.
        // Every bin accumulates the same ordered path subsequence, so the
        // final FP64 reduction tree is also invariant under team tuning.
        double sums[TeamSize]{};
        double sums_squared[TeamSize]{};
        const std::size_t rounds =
            1U + (paths_per_price - 1U) /
                (static_cast<std::size_t>(blockDim.x) * gridDim.y);
        for (std::size_t round = 0U; round < rounds; ++round) {
            #pragma unroll
            for (unsigned int slot = 0U; slot < TeamSize; ++slot) {
                const std::size_t path =
                    (round * gridDim.y + blockIdx.y) * blockDim.x
                    + slot * teams_per_block + team;
                if (path >= paths_per_price) continue;
                float value = 0.0f;
                if constexpr (TeamSize == 1U) {
                    typename Pack::States states;
                    value = evaluate_multi_asset_path<Pack, Product, TeamSize>(
                        states, prepared, prepared_product,
                        product_parameters, lower, nullptr,
                        key, path, steps,
                        simulation_steps_per_day, lane
                    );
                } else {
                    value = evaluate_multi_asset_path<Pack, Product, TeamSize>(
                        team_states[team], prepared, prepared_product,
                        product_parameters, lower,
                        use_shared_normals
                            ? team_innovations+team*Pack::kFactorCount
                            : nullptr,
                        key, path, steps,
                        simulation_steps_per_day, lane
                    );
                }
                if (lane == 0U) {
                    const double payoff = static_cast<double>(value);
                    sums[slot] += payoff;
                    sums_squared[slot] += payoff * payoff;
                }
            }
        }

        // All path teams must finish before their shared factor and state
        // storage is reused for the fixed logical-bin moments.
        __syncthreads();
        auto* bin_sums = reinterpret_cast<double*>(workspace);
        auto* bin_squares = bin_sums + blockDim.x;
        if (lane == 0U) {
            #pragma unroll
            for (unsigned int slot = 0U; slot < TeamSize; ++slot) {
                const unsigned int bin = slot * teams_per_block + team;
                bin_sums[bin] = sums[slot];
                bin_squares[bin] = sums_squared[slot];
            }
        }
        __syncthreads();
        const double sum = bin_sums[threadIdx.x];
        const double sumsq = bin_squares[threadIdx.x];
        const auto moments = reductions::reduce_block(sum, sumsq);
        if (threadIdx.x == 0U) {
            if (gridDim.y == 1U) {
                double price = 0.0;
                double error = 0.0;
                reductions::compute_statistics(
                    moments, paths_per_price, price, error, rounds
                );
                prices[result_index] = static_cast<float>(price);
                standard_errors[result_index] = static_cast<float>(error);
            } else {
                partial_moments[launch_index * gridDim.y + blockIdx.y] =
                    moments;
            }
        }
        __syncthreads();
    }
}

template<typename Pack, typename Product, unsigned int TeamSize>
inline void launch_multi_asset_cuda(
    MultiAssetDeviceInputs<Pack, Product> inputs,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t paths_per_price,
    float dt,
    std::uint32_t steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* prices,
    float* standard_errors,
    std::size_t blocks_per_price = 1U,
    reductions::MomentSums* partial_moments = nullptr
) {
    static_assert(Pack::kFactorCount <= kMaximumSharedCholeskyFactors,
        "Multi-asset Brownian dimension exceeds the shared-Cholesky factor limit.");
    if (result_count == 0U || launch_result_count == 0U
        || result_offset >= result_count
        || launch_result_count > result_count - result_offset
        || paths_per_price < 2U || !(dt > 0.0f)
        || steps_per_day == 0U || block_count == 0U
        || blocks_per_price == 0U || blocks_per_price > 65535U
        || (blocks_per_price > 1U && partial_moments == nullptr)
        || threads_per_block < 32U || threads_per_block > 1024U
        || threads_per_block % 32U != 0U
        || threads_per_block % TeamSize != 0U) {
        throw std::invalid_argument("Invalid multi-asset MC launch geometry.");
    }
    validate_device_pointer(inputs.products, "multi-asset products");
    validate_device_pointer(
        inputs.row_selectors, "multi-asset row selectors");
    validate_device_pointer(inputs.packed_factors, "multi-asset factors");
    validate_device_pointer(prices, "multi-asset prices");
    validate_device_pointer(standard_errors, "multi-asset standard errors");
    if (blocks_per_price > 1U)
        validate_device_pointer(partial_moments, "multi-asset moments");
    constexpr std::size_t lower_count =
        Pack::kFactorCount * (Pack::kFactorCount + 1U) / 2U;
    const std::size_t reduction_bytes =
        2U * (threads_per_block / 32U) * sizeof(double);
    const std::size_t lower_offset =
        align_workspace(reduction_bytes, alignof(float));
    const std::size_t state_offset = align_workspace(
        lower_offset + lower_count * sizeof(float),
        alignof(typename Pack::States)
    );
    const std::size_t path_storage_bytes = align_workspace(
        state_offset
            + (TeamSize == 1U ? 0U : threads_per_block / TeamSize)
                * sizeof(typename Pack::States),
        alignof(float));
    const std::size_t moment_bytes =
        2U * threads_per_block * sizeof(double);
    const std::size_t base_shared_bytes=
        path_storage_bytes>moment_bytes
            ? path_storage_bytes : moment_bytes;
    const std::size_t normal_bytes=TeamSize>1U
        ? Pack::kFactorCount*sizeof(float)
            *(threads_per_block/TeamSize) : 0U;
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(
        &attributes, multi_asset_price_kernel<Pack, Product, TeamSize>
    ), "multi-asset kernel attributes");
    int device = 0;
    check_cuda(cudaGetDevice(&device), "multi-asset CUDA device");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, device),
        "multi-asset device properties");
    const bool use_shared_normals=normal_bytes>0U
        && base_shared_bytes+normal_bytes+attributes.sharedSizeBytes
            <=properties.sharedMemPerBlockOptin;
    const std::size_t shared_bytes=base_shared_bytes
        +(use_shared_normals ? normal_bytes : 0U);
    if (shared_bytes + attributes.sharedSizeBytes
        > properties.sharedMemPerBlockOptin) {
        throw std::invalid_argument(
            "Multi-asset block exceeds available shared memory."
        );
    }
    if (shared_bytes + attributes.sharedSizeBytes
        > properties.sharedMemPerBlock) {
        check_cuda(cudaFuncSetAttribute(
            multi_asset_price_kernel<Pack, Product, TeamSize>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)
        ), "multi-asset shared memory opt-in");
    }
    multi_asset_price_kernel<Pack, Product, TeamSize>
        <<<dim3(static_cast<unsigned int>(block_count),
                  static_cast<unsigned int>(blocks_per_price)),
           threads_per_block, shared_bytes>>>(
            inputs, result_offset, launch_result_count, paths_per_price,
            dt, steps_per_day, base_seed, prices, standard_errors,
            partial_moments, use_shared_normals
        );
    check_cuda(cudaGetLastError(), "multi-asset price kernel");
}

}  // namespace ai_factory::workbench::monte_carlo
