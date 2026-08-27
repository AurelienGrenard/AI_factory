// Kappa=1 hybrid discretization of a normalized fractional Volterra driver.
#pragma once

#include "common/volterra/fractional_kernel.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace ai_factory::workbench::volterra {

// The simulated process is
//
//   Y_t = sqrt(2H) integral_0^t (t-s)^(H-1/2) dW_s,
//
// so Var(Y_t) = t^(2H). The current singular cell is sampled exactly and
// older cells use the cell-average power-kernel weights of the hybrid scheme.
struct FractionalHybridDriverPolicy {
    using Parameters = float;

    struct PreparedDriver {
        FractionalPowerKernel kernel;
        float time_step;
        float sqrt_time_step;
        float time_step_to_exponent;
        float sqrt_two_h;
        float singular_rough_loading;
        float singular_independent_loading;
    };

    __host__ __device__ static PreparedDriver prepare(
        Parameters hurst_exponent,
        float time_step
    ) {
        const FractionalPowerKernel kernel =
            FractionalPowerKernel::prepare(hurst_exponent);
        const float time_step_to_h = powf(time_step, hurst_exponent);
        const float singular_rough_loading =
            time_step_to_h / kernel.exponent_plus_one;
        const float singular_variance =
            time_step_to_h * time_step_to_h / kernel.two_h;
        const float independent_variance = fmaxf(
            singular_variance
                - singular_rough_loading * singular_rough_loading,
            0.0f
        );
        return {
            kernel,
            time_step,
            sqrtf(time_step),
            powf(time_step, kernel.exponent),
            sqrtf(kernel.two_h),
            singular_rough_loading,
            sqrtf(independent_variance),
        };
    }

    __host__ __device__ static float far_cell_weight(
        const PreparedDriver& driver,
        unsigned int lag
    ) {
        return driver.kernel.cell_average_weight_from_scale(
            driver.time_step_to_exponent,
            lag
        );
    }

    __host__ __device__ static float variance(
        const PreparedDriver& driver,
        float time
    ) {
        return driver.kernel.normalized_variance(time);
    }

    __host__ __device__ static float value(
        const PreparedDriver& driver,
        float far_convolution,
        float rough_normal,
        float singular_independent_normal
    ) {
        const float singular_integral = fmaf(
            driver.singular_rough_loading,
            rough_normal,
            driver.singular_independent_loading
                * singular_independent_normal
        );
        return driver.sqrt_two_h * (singular_integral + far_convolution);
    }
};

static_assert(std::is_trivially_copyable_v<
    FractionalHybridDriverPolicy::PreparedDriver
>);

}  // namespace ai_factory::workbench::volterra
