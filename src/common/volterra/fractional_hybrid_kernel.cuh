// Kappa=1 hybrid discretization of a normalized fractional Volterra kernel.
#pragma once

#include "common/volterra/concepts.cuh"
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
struct FractionalHybridKernelPolicy {
    using Parameters = float;

    struct PreparedKernel {
        FractionalPowerKernel kernel;
        float time_step;
        float time_step_to_exponent;
        float sqrt_two_h;
        float singular_rough_loading;
        float singular_independent_loading;
    };

    __host__ __device__ static PreparedKernel prepare(
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
            powf(time_step, kernel.exponent),
            sqrtf(kernel.two_h),
            singular_rough_loading,
            sqrtf(independent_variance),
        };
    }

    __host__ __device__ static float far_cell_weight(
        const PreparedKernel& kernel,
        unsigned int lag
    ) {
        return kernel.kernel.cell_average_weight_from_scale(
            kernel.time_step_to_exponent,
            lag
        );
    }

    __host__ __device__ static float volterra_variance(
        const PreparedKernel& kernel,
        float time_years
    ) {
        return kernel.kernel.normalized_variance(time_years);
    }

    __host__ __device__ static float reconstruct_volterra_value(
        const PreparedKernel& kernel,
        float far_convolution,
        float rough_normal,
        float singular_independent_normal
    ) {
        const float singular_integral = fmaf(
            kernel.singular_rough_loading,
            rough_normal,
            kernel.singular_independent_loading
                * singular_independent_normal
        );
        return kernel.sqrt_two_h * (singular_integral + far_convolution);
    }
};

static_assert(HybridKernelPolicy<FractionalHybridKernelPolicy>);

}  // namespace ai_factory::workbench::volterra
