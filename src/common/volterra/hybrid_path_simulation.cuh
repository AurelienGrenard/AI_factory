// Shared Volterra path traversal; the caller owns observations and convolution storage.
#pragma once
#include "common/philox.cuh"
#include "common/volterra/hybrid_fft.cuh"
#include <cuda_runtime.h>

namespace ai_factory::workbench::volterra::hybrid_fft {

struct FftPathConvolution {
    const float2* values;
    std::size_t local_pair;
    bool imaginary_lane;
    template<typename Row>
    __device__ __forceinline__ float value(const Row& row, std::size_t, std::uint32_t step) const {
        if (step == 0U) return 0.0f;
        const float2 packed = values[local_pair * row.schedule.step_count + step - 1U];
        return imaginary_lane ? packed.y : packed.x;
    }
};

struct DirectPathConvolution {
    template<typename Row>
    __device__ __forceinline__ float value(const Row& row, std::size_t path, std::uint32_t step) const {
        using Kernel = typename Row::Kernel;
        float result = 0.0f;
        for (std::uint32_t prior = 0U; prior < step; ++prior) {
            const float increment = row.sqrt_time_step * normal_at(row.key, path, 3ULL * prior);
            result += increment * Kernel::far_cell_weight(row.kernel, step - prior + 1U);
        }
        return result;
    }
};

template<typename Dynamics, typename Row, typename Convolution, typename Observer>
__device__ __forceinline__ typename Dynamics::State simulate_observed_path(
    const Row& row, const typename Dynamics::PreparedModel& model, std::size_t path,
    const float* volterra_variances, Convolution convolution, Observer& observation_adapter
) {
    using Schedule = typename Row::Schedule;
    using Kernel = typename Row::Kernel;
    typename Dynamics::State state = Dynamics::initial_state(model);
    typename Schedule::Cursor cursor =
        Schedule::make_cursor(row.schedule);
    bool keep_running = Schedule::on_initial_state(
        row.schedule,
        cursor,
        state,
        observation_adapter
    );
    philox::UniformSequence uniforms(row.key, path);
    philox::NormalPairCache normal_cache;
    for (std::uint32_t step = 0U;
         keep_running && step < row.schedule.step_count;
         ++step) {
        const float rough_normal =
            philox::next_normal(uniforms, normal_cache);
        const float singular_normal =
            philox::next_normal(uniforms, normal_cache);
        const float spot_normal =
            philox::next_normal(uniforms, normal_cache);
        const float far_convolution = convolution.value(row, path, step);
        const float volterra_value =
            Kernel::reconstruct_volterra_value(
                row.kernel,
                far_convolution,
                rough_normal,
                singular_normal
            );
        Dynamics::advance(
            model,
            volterra_value,
            volterra_variances[step],
            rough_normal,
            spot_normal,
            state
        );
        keep_running = Schedule::on_step(
            row.schedule,
            cursor,
            step,
            state,
            observation_adapter
        );
    }
    return state;
}

}  // namespace ai_factory::workbench::volterra::hybrid_fft
