#pragma once

namespace ai_factory::cuda::reductions {

template <int ValueCount>
__device__ __forceinline__ void reduce_block_values(double (&values)[ValueCount]) {
    extern __shared__ double shared[];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int warp_count = (blockDim.x + 31U) >> 5U;
    const unsigned int mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int value = 0; value < ValueCount; ++value) {
            values[value] += __shfl_down_sync(mask, values[value], offset);
        }
    }
    if (lane == 0U) {
#pragma unroll
        for (int value = 0; value < ValueCount; ++value) {
            shared[value * blockDim.x + warp] = values[value];
        }
    }
    __syncthreads();
    if (warp == 0U) {
#pragma unroll
        for (int value = 0; value < ValueCount; ++value) {
            values[value] = lane < warp_count
                                ? shared[value * blockDim.x + lane]
                                : 0.0;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int value = 0; value < ValueCount; ++value) {
                values[value] +=
                    __shfl_down_sync(0xFFFFFFFFU, values[value], offset);
            }
        }
        if (lane == 0U) {
#pragma unroll
            for (int value = 0; value < ValueCount; ++value) {
                shared[value * blockDim.x] = values[value];
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void reduce_block(
    double local_sum,
    double local_sumsq
) {
    extern __shared__ double shared[];
    double* sums = shared;
    double* sumsq = shared + blockDim.x;
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int warp_count = (blockDim.x + 31U) >> 5U;
    const unsigned int mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
        local_sumsq += __shfl_down_sync(mask, local_sumsq, offset);
    }
    if (lane == 0U) {
        sums[warp] = local_sum;
        sumsq[warp] = local_sumsq;
    }
    __syncthreads();

    if (warp == 0U) {
        local_sum = lane < warp_count ? sums[lane] : 0.0;
        local_sumsq = lane < warp_count ? sumsq[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFFU, local_sum, offset);
            local_sumsq += __shfl_down_sync(
                0xFFFFFFFFU,
                local_sumsq,
                offset
            );
        }
        if (lane == 0U) {
            sums[0] = local_sum;
            sumsq[0] = local_sumsq;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void reduce_block_four(
    double first,
    double second,
    double third,
    double fourth
) {
    extern __shared__ double shared[];
    double* first_values = shared;
    double* second_values = shared + blockDim.x;
    double* third_values = shared + 2U * blockDim.x;
    double* fourth_values = shared + 3U * blockDim.x;
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;
    const unsigned int warp_count = (blockDim.x + 31U) >> 5U;
    const unsigned int mask = __activemask();
    for (int offset = 16; offset > 0; offset >>= 1) {
        first += __shfl_down_sync(mask, first, offset);
        second += __shfl_down_sync(mask, second, offset);
        third += __shfl_down_sync(mask, third, offset);
        fourth += __shfl_down_sync(mask, fourth, offset);
    }
    if (lane == 0U) {
        first_values[warp] = first;
        second_values[warp] = second;
        third_values[warp] = third;
        fourth_values[warp] = fourth;
    }
    __syncthreads();

    if (warp == 0U) {
        first = lane < warp_count ? first_values[lane] : 0.0;
        second = lane < warp_count ? second_values[lane] : 0.0;
        third = lane < warp_count ? third_values[lane] : 0.0;
        fourth = lane < warp_count ? fourth_values[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            first += __shfl_down_sync(0xFFFFFFFFU, first, offset);
            second += __shfl_down_sync(0xFFFFFFFFU, second, offset);
            third += __shfl_down_sync(0xFFFFFFFFU, third, offset);
            fourth += __shfl_down_sync(0xFFFFFFFFU, fourth, offset);
        }
        if (lane == 0U) {
            first_values[0] = first;
            second_values[0] = second;
            third_values[0] = third;
            fourth_values[0] = fourth;
        }
    }
    __syncthreads();
}

}  // namespace ai_factory::cuda::reductions
