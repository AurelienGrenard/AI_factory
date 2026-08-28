// Hybrid discretization for the normalized log-modulated fractional kernel.
#pragma once

#include "common/volterra/concepts.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace ai_factory::workbench::volterra {

// Bayer-Harang-Pigato kernel
//
//   K(t) = C t^(H-1/2) max(zeta log(1/t), 1)^(-p),
//
// where C is chosen so integral_0^1 K(t)^2 dt = 1.  Kernel integrals are
// prepared once per result row; the path kernels only consume their FFT.
struct LogModulatedHybridKernelPolicy {
    struct Parameters {
        float hurst_exponent;
        float log_modulation_scale;
        float log_modulation_power;
    };

    struct PreparedKernel {
        Parameters parameters;
        float normalization;
        float time_step;
        float singular_rough_loading;
        float singular_independent_loading;
    };

    __host__ __device__ static float modulation(
        const Parameters& parameters,
        float time
    ) {
        const float logarithm = logf(1.0f / time);
        return powf(
            fmaxf(parameters.log_modulation_scale * logarithm, 1.0f),
            -parameters.log_modulation_power
        );
    }

    __host__ __device__ static float unnormalized_kernel(
        const Parameters& parameters,
        float time
    ) {
        return powf(time, parameters.hurst_exponent - 0.5f)
            * modulation(parameters, time);
    }

    // Integrate K_unscaled^q from zero to upper.  The logarithmic change of
    // variable removes the singularity.  A rational map covers the infinite
    // tail with a fixed, deterministic quadrature used only during row prep.
    __host__ __device__ static float unnormalized_power_integral(
        const Parameters& parameters,
        float upper,
        float power
    ) {
        if (!(upper > 0.0f)) return 0.0f;
        const float lower_x = -logf(upper);
        const float threshold = 1.0f / parameters.log_modulation_scale;
        const float rate = power * (parameters.hurst_exponent - 0.5f) + 1.0f;
        float integral = 0.0f;
        float tail_start = lower_x;
        if (lower_x < threshold) {
            integral = fabsf(rate) < 1.0e-7f
                ? threshold - lower_x
                : (expf(-rate * lower_x) - expf(-rate * threshold))
                    / rate;
            tail_start = threshold;
        }

        constexpr int quadrature_points = 96;
        // Match the rational map to the logarithmic slope at the start of
        // the tail.  Scaling only with `rate` badly under-resolves the narrow
        // boundary layer when H is close to zero, zeta is large and p is
        // large (the stress region of the parameter dataset).
        const float local_decay = rate
            + power * parameters.log_modulation_power / tail_start;
        const float scale = 1.0f / fmaxf(local_decay, 1.0e-6f);
        float tail = 0.0f;
        for (int point = 0; point < quadrature_points; ++point) {
            const float u = (static_cast<float>(point) + 0.5f)
                / static_cast<float>(quadrature_points);
            const float one_minus_u = 1.0f - u;
            const float x = tail_start + scale * u / one_minus_u;
            const float jacobian = scale / (one_minus_u * one_minus_u);
            tail += expf(-rate * x)
                * powf(
                    parameters.log_modulation_scale * x,
                    -power * parameters.log_modulation_power
                )
                * jacobian;
        }
        return integral + tail / static_cast<float>(quadrature_points);
    }

    __host__ __device__ static PreparedKernel prepare(
        const Parameters& parameters,
        float time_step
    ) {
        const float squared_norm = unnormalized_power_integral(
            parameters, 1.0f, 2.0f
        );
        const float normalization = 1.0f / sqrtf(squared_norm);
        const float first_moment = normalization
            * unnormalized_power_integral(parameters, time_step, 1.0f);
        const float second_moment = normalization * normalization
            * unnormalized_power_integral(parameters, time_step, 2.0f);
        const float rough_loading = first_moment / sqrtf(time_step);
        return {
            parameters,
            normalization,
            time_step,
            rough_loading,
            sqrtf(fmaxf(
                second_moment - rough_loading * rough_loading,
                0.0f
            )),
        };
    }

    __host__ __device__ static float far_cell_weight(
        const PreparedKernel& kernel,
        unsigned int lag
    ) {
        const float lower = static_cast<float>(lag - 1U) * kernel.time_step;
        const float upper = static_cast<float>(lag) * kernel.time_step;
        constexpr int intervals = 8;
        const float h = (upper - lower) / static_cast<float>(intervals);
        float sum = unnormalized_kernel(kernel.parameters, lower)
            + unnormalized_kernel(kernel.parameters, upper);
        for (int index = 1; index < intervals; ++index) {
            sum += (index & 1 ? 4.0f : 2.0f)
                * unnormalized_kernel(
                    kernel.parameters,
                    fmaf(static_cast<float>(index), h, lower)
                );
        }
        return kernel.normalization * sum * h
            / (3.0f * kernel.time_step);
    }

    __host__ __device__ static float volterra_variance(
        const PreparedKernel& kernel,
        float time
    ) {
        return kernel.normalization * kernel.normalization
            * unnormalized_power_integral(kernel.parameters, time, 2.0f);
    }

    __host__ __device__ static float reconstruct_volterra_value(
        const PreparedKernel& kernel,
        float far_convolution,
        float rough_normal,
        float singular_independent_normal
    ) {
        return far_convolution
            + kernel.singular_rough_loading * rough_normal
            + kernel.singular_independent_loading
                * singular_independent_normal;
    }
};

static_assert(HybridKernelPolicy<LogModulatedHybridKernelPolicy>);

}  // namespace ai_factory::workbench::volterra
