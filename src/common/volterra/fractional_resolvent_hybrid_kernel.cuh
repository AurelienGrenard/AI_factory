// Hybrid discretization of a mean-reverting fractional Gaussian resolvent.
#pragma once

#include "common/compensated_sum.cuh"
#include "common/volterra/concepts.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace ai_factory::workbench::volterra {

// If K_H(t)=sqrt(2H)t^(H-1/2), the stochastic solution of
// X = -kappa K_H * X dt + K_H * dW uses the resolvent
// R(t)=A t^(alpha-1) E_(alpha,alpha)(-kappa A t^alpha),
// alpha=H+1/2 and A=sqrt(2H) Gamma(alpha).
struct FractionalResolventHybridKernelPolicy {
    struct Parameters {
        float hurst_exponent;
        float mean_reversion;
    };

    struct PreparedKernel {
        Parameters parameters;
        float alpha;
        float laplace_scale;
        float time_step;
        float singular_rough_loading;
        float singular_independent_loading;
    };

    __host__ __device__ __noinline__ static float
    mittag_leffler_alpha_alpha(
        float alpha,
        float argument
    ) {
        // The power series is accurate near the origin but suffers severe
        // cancellation on the negative real axis in FP32.  From x=2 onward,
        // use the positive Laplace-density representation of E_a(-x),
        // together with E_(a,a)(-x)=-a d E_a(-x)/dx.  The crossover and the
        // compensated sums are qualified over the published H/kappa/time
        // domain by fractional_resolvent_precision_cuda_test.cu.
        const float x = -argument;
        constexpr float crossover = 2.0f;
        if (x > crossover) {
            constexpr int quadrature_points = 96;
            constexpr float pi = 3.14159265358979323846f;
            const float transformed_time = powf(x, 1.0f / alpha);
            const float prefactor = powf(x, 1.0f / alpha - 1.0f)
                / (transformed_time * transformed_time);
            const float sine_scale = sinf(pi * alpha) / pi;
            const float cosine = cosf(pi * alpha);
            CompensatedFloatSum sum;
            #pragma unroll 1
            for (int point = 0; point < quadrature_points; ++point) {
                const float u = (static_cast<float>(point) + 0.5f)
                    / static_cast<float>(quadrature_points);
                const float one_minus_u = 1.0f - u;
                const float y = u / one_minus_u;
                const float r = y / transformed_time;
                const float r_to_alpha = powf(r, alpha);
                const float density = sine_scale * powf(r, alpha - 1.0f)
                    / (r_to_alpha * r_to_alpha
                       + 2.0f * r_to_alpha * cosine + 1.0f);
                sum.add(
                    y * expf(-y) * density
                    / (one_minus_u * one_minus_u)
                );
            }
            return prefactor * sum.value()
                / static_cast<float>(quadrature_points);
        }
        CompensatedFloatSum sum;
        float power = 1.0f;
        #pragma unroll 1
        for (int order = 0; order < 96; ++order) {
            const float term = power / tgammaf(alpha * order + alpha);
            sum.add(term);
            if (order > 8
                && fabsf(term) < 2.0e-6f * fmaxf(fabsf(sum.value()), 1.0f)) {
                break;
            }
            power *= argument;
        }
        return sum.value();
    }

    __host__ __device__ __noinline__ static float kernel(
        const PreparedKernel& kernel,
        float time_years
    ) {
        const float argument = -kernel.parameters.mean_reversion
            * kernel.laplace_scale * powf(time_years, kernel.alpha);
        return kernel.laplace_scale
            * powf(time_years, kernel.alpha - 1.0f)
            * mittag_leffler_alpha_alpha(kernel.alpha, argument);
    }

    __host__ __device__ __noinline__ static float power_integral(
        const PreparedKernel& kernel,
        float upper,
        int power
    ) {
        if (!(upper > 0.0f)) return 0.0f;
        const float lower_x = -logf(upper);
        const float rate = power == 2
            ? 2.0f * kernel.parameters.hurst_exponent
            : kernel.alpha;
        constexpr int quadrature_points = 96;
        const float scale = 1.0f / fmaxf(rate, 0.02f);
        CompensatedFloatSum sum;
        #pragma unroll 1
        for (int point = 0; point < quadrature_points; ++point) {
            const float u = (static_cast<float>(point) + 0.5f)
                / static_cast<float>(quadrature_points);
            const float one_minus_u = 1.0f - u;
            const float x = lower_x + scale * u / one_minus_u;
            float integrand = 0.0f;
            if (x > 70.0f) {
                const float leading = sqrtf(
                    2.0f * kernel.parameters.hurst_exponent
                );
                integrand = (power == 2 ? leading * leading : leading)
                    * expf(-rate * x);
            } else {
                const float time_years = expf(-x);
                const float value =
                    FractionalResolventHybridKernelPolicy::kernel(
                        kernel,
                        time_years
                    );
                const float powered = power == 2 ? value * value : value;
                integrand = powered * time_years;
            }
            sum.add(integrand * scale / (one_minus_u * one_minus_u));
        }
        return sum.value() / static_cast<float>(quadrature_points);
    }

    __host__ __device__ __noinline__ static PreparedKernel prepare(
        const Parameters& parameters,
        float time_step
    ) {
        PreparedKernel kernel{
            parameters,
            parameters.hurst_exponent + 0.5f,
            0.0f,
            time_step,
            0.0f,
            0.0f,
        };
        kernel.laplace_scale = sqrtf(2.0f * parameters.hurst_exponent)
            * tgammaf(kernel.alpha);
        const float first_moment = power_integral(kernel, time_step, 1);
        const float second_moment = power_integral(kernel, time_step, 2);
        kernel.singular_rough_loading = first_moment / sqrtf(time_step);
        kernel.singular_independent_loading = sqrtf(fmaxf(
            second_moment
                - kernel.singular_rough_loading
                    * kernel.singular_rough_loading,
            0.0f
        ));
        return kernel;
    }

    __host__ __device__ __noinline__ static float far_cell_weight(
        const PreparedKernel& kernel,
        unsigned int lag
    ) {
        const float lower = static_cast<float>(lag - 1U) * kernel.time_step;
        const float upper = static_cast<float>(lag) * kernel.time_step;
        constexpr int intervals = 8;
        const float h = (upper - lower) / static_cast<float>(intervals);
        CompensatedFloatSum sum;
        sum.add(FractionalResolventHybridKernelPolicy::kernel(kernel, lower));
        sum.add(FractionalResolventHybridKernelPolicy::kernel(kernel, upper));
        for (int index = 1; index < intervals; ++index) {
            sum.add((index & 1 ? 4.0f : 2.0f)
                * FractionalResolventHybridKernelPolicy::kernel(
                    kernel,
                    fmaf(static_cast<float>(index), h, lower)
                ));
        }
        return sum.value() * h / (3.0f * kernel.time_step);
    }

    __host__ __device__ __noinline__ static float volterra_variance(
        const PreparedKernel& kernel,
        float time_years
    ) {
        return power_integral(kernel, time_years, 2);
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

static_assert(HybridKernelPolicy<FractionalResolventHybridKernelPolicy>);

}  // namespace ai_factory::workbench::volterra
