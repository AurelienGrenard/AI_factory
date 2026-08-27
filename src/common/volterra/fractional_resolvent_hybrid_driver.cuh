// Hybrid discretization of a mean-reverting fractional Gaussian resolvent.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace ai_factory::workbench::volterra {

// If K_H(t)=sqrt(2H)t^(H-1/2), the stochastic solution of
// X = -kappa K_H * X dt + K_H * dW uses the resolvent
// R(t)=A t^(alpha-1) E_(alpha,alpha)(-kappa A t^alpha),
// alpha=H+1/2 and A=sqrt(2H) Gamma(alpha).
struct FractionalResolventHybridDriverPolicy {
    struct Parameters {
        float hurst_exponent;
        float mean_reversion;
    };

    struct PreparedDriver {
        Parameters parameters;
        float alpha;
        float laplace_scale;
        float time_step;
        float sqrt_time_step;
        float singular_rough_loading;
        float singular_independent_loading;
    };

    __host__ __device__ __noinline__ static double
    mittag_leffler_alpha_alpha(
        double alpha,
        double argument
    ) {
        // The power series is accurate near the origin but suffers severe
        // cancellation on the negative real axis.  Beyond the crossover use
        // the positive Laplace-density representation of E_a(-x), together
        // with E_(a,a)(-x)=-a d E_a(-x)/dx.  Scaling r by x^(1/a) keeps the
        // quadrature concentrated around order-one values even in the tail.
        const double x = -argument;
        const double crossover = 3.5 + 12.0 * (alpha - 0.5);
        if (x > crossover) {
            constexpr int quadrature_points = 96;
            constexpr double pi = 3.141592653589793238462643383279502884;
            const double transformed_time = pow(x, 1.0 / alpha);
            const double prefactor = pow(x, 1.0 / alpha - 1.0)
                / (transformed_time * transformed_time);
            const double sine_scale = sin(pi * alpha) / pi;
            const double cosine = cos(pi * alpha);
            double sum = 0.0;
            #pragma unroll 1
            for (int point = 0; point < quadrature_points; ++point) {
                const double u = (static_cast<double>(point) + 0.5)
                    / static_cast<double>(quadrature_points);
                const double one_minus_u = 1.0 - u;
                const double y = u / one_minus_u;
                const double r = y / transformed_time;
                const double r_to_alpha = pow(r, alpha);
                const double density = sine_scale * pow(r, alpha - 1.0)
                    / (r_to_alpha * r_to_alpha
                       + 2.0 * r_to_alpha * cosine + 1.0);
                sum += y * exp(-y) * density
                    / (one_minus_u * one_minus_u);
            }
            return prefactor * sum
                / static_cast<double>(quadrature_points);
        }
        double sum = 0.0;
        double power = 1.0;
        #pragma unroll 1
        for (int order = 0; order < 96; ++order) {
            const double term = power / tgamma(alpha * order + alpha);
            sum += term;
            if (order > 8 && fabs(term) < 2.0e-14 * fmax(fabs(sum), 1.0)) {
                break;
            }
            power *= argument;
        }
        return sum;
    }

    __host__ __device__ __noinline__ static float kernel(
        const PreparedDriver& driver,
        float time
    ) {
        const double alpha = driver.alpha;
        const double t = time;
        const double argument = -static_cast<double>(
            driver.parameters.mean_reversion * driver.laplace_scale
        ) * pow(t, alpha);
        return static_cast<float>(
            static_cast<double>(driver.laplace_scale)
            * pow(t, alpha - 1.0)
            * mittag_leffler_alpha_alpha(alpha, argument)
        );
    }

    __host__ __device__ __noinline__ static float power_integral(
        const PreparedDriver& driver,
        float upper,
        int power
    ) {
        if (!(upper > 0.0f)) return 0.0f;
        const float lower_x = -logf(upper);
        const float rate = power == 2
            ? 2.0f * driver.parameters.hurst_exponent
            : driver.alpha;
        constexpr int quadrature_points = 96;
        const float scale = 1.0f / fmaxf(rate, 0.02f);
        double sum = 0.0;
        #pragma unroll 1
        for (int point = 0; point < quadrature_points; ++point) {
            const float u = (static_cast<float>(point) + 0.5f)
                / static_cast<float>(quadrature_points);
            const float one_minus_u = 1.0f - u;
            const float x = lower_x + scale * u / one_minus_u;
            double integrand = 0.0;
            if (x > 70.0f) {
                const double leading = sqrt(
                    2.0 * driver.parameters.hurst_exponent
                );
                integrand = (power == 2 ? leading * leading : leading)
                    * exp(-static_cast<double>(rate) * x);
            } else {
                const float time = expf(-x);
                const double value = kernel(driver, time);
                const double powered = power == 2 ? value * value : value;
                integrand = powered * static_cast<double>(time);
            }
            sum += integrand * scale
                / static_cast<double>(one_minus_u * one_minus_u);
        }
        return static_cast<float>(sum / quadrature_points);
    }

    __host__ __device__ __noinline__ static PreparedDriver prepare(
        const Parameters& parameters,
        float time_step
    ) {
        PreparedDriver driver{
            parameters,
            parameters.hurst_exponent + 0.5f,
            0.0f,
            time_step,
            sqrtf(time_step),
            0.0f,
            0.0f,
        };
        driver.laplace_scale = sqrtf(2.0f * parameters.hurst_exponent)
            * tgammaf(driver.alpha);
        const float first_moment = power_integral(driver, time_step, 1);
        const float second_moment = power_integral(driver, time_step, 2);
        driver.singular_rough_loading = first_moment / driver.sqrt_time_step;
        driver.singular_independent_loading = sqrtf(fmaxf(
            second_moment
                - driver.singular_rough_loading
                    * driver.singular_rough_loading,
            0.0f
        ));
        return driver;
    }

    __host__ __device__ __noinline__ static float far_cell_weight(
        const PreparedDriver& driver,
        unsigned int lag
    ) {
        const float lower = static_cast<float>(lag - 1U) * driver.time_step;
        const float upper = static_cast<float>(lag) * driver.time_step;
        constexpr int intervals = 8;
        const float h = (upper - lower) / static_cast<float>(intervals);
        float sum = kernel(driver, lower) + kernel(driver, upper);
        for (int index = 1; index < intervals; ++index) {
            sum += (index & 1 ? 4.0f : 2.0f) * kernel(
                driver,
                fmaf(static_cast<float>(index), h, lower)
            );
        }
        return sum * h / (3.0f * driver.time_step);
    }

    __host__ __device__ __noinline__ static float variance(
        const PreparedDriver& driver,
        float time
    ) {
        return power_integral(driver, time, 2);
    }

    __host__ __device__ static float value(
        const PreparedDriver& driver,
        float far_convolution,
        float rough_normal,
        float singular_independent_normal
    ) {
        return far_convolution
            + driver.singular_rough_loading * rough_normal
            + driver.singular_independent_loading
                * singular_independent_normal;
    }
};

static_assert(std::is_trivially_copyable_v<
    FractionalResolventHybridDriverPolicy::PreparedDriver
>);

}  // namespace ai_factory::workbench::volterra
