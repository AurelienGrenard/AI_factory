// Exercise host validation and finite-coefficient guarantees for quadratic rough-Heston lifts.
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {

namespace quadratic =
    ai_factory::workbench::model::equity::quadratic_rough_heston;
namespace rough_heston =
    ai_factory::workbench::model::equity::rough_heston;
namespace volterra = ai_factory::workbench::volterra;

using QuadraticParameters = quadratic::ModelParameters;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename Exception = std::invalid_argument, typename Function>
void require_throws(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

QuadraticParameters valid_model() {
    return {
        1.0f,
        0.02f,
        0.01f,
        0.10f,
        0.50f,
        0.05f,
        0.001f,
        1.0f,
        0.50f,
        0.10f,
    };
}

volterra::ExponentialKernel<2U> valid_kernel() {
    return {{1.0f, 4.0f}, {0.5f, 0.25f}};
}

void require_invalid_model_value(
    float QuadraticParameters::* member,
    float value,
    const char* message
) {
    QuadraticParameters model = valid_model();
    model.*member = value;
    const auto kernel = valid_kernel();
    require_throws([&] {
        (void)quadratic::prepare_dynamics(model, kernel, 1.0f / 252.0f);
    }, message);
}

}  // namespace

int main() {
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float infinity = std::numeric_limits<float>::infinity();

    constexpr float QuadraticParameters::* all_members[] = {
        &QuadraticParameters::spot,
        &QuadraticParameters::risk_free_rate,
        &QuadraticParameters::dividend_yield,
        &QuadraticParameters::initial_feedback,
        &QuadraticParameters::quadratic_scale,
        &QuadraticParameters::quadratic_shift,
        &QuadraticParameters::variance_floor,
        &QuadraticParameters::feedback_rate,
        &QuadraticParameters::feedback_volatility,
        &QuadraticParameters::hurst_exponent,
    };
    for (float QuadraticParameters::* member : all_members) {
        require_invalid_model_value(
            member,
            nan,
            "Quadratic rough-Heston accepted a NaN model input."
        );
        require_invalid_model_value(
            member,
            infinity,
            "Quadratic rough-Heston accepted an infinite model input."
        );
    }

    constexpr float QuadraticParameters::* positive_members[] = {
        &QuadraticParameters::spot,
        &QuadraticParameters::quadratic_scale,
        &QuadraticParameters::variance_floor,
        &QuadraticParameters::feedback_rate,
        &QuadraticParameters::feedback_volatility,
        &QuadraticParameters::hurst_exponent,
    };
    for (float QuadraticParameters::* member : positive_members) {
        require_invalid_model_value(
            member,
            0.0f,
            "Quadratic rough-Heston accepted a zero positive-domain input."
        );
        require_invalid_model_value(
            member,
            -1.0f,
            "Quadratic rough-Heston accepted a negative positive-domain input."
        );
    }
    require_invalid_model_value(
        &QuadraticParameters::hurst_exponent,
        0.5f,
        "Quadratic rough-Heston accepted H=0.5."
    );

    for (const float invalid_dt : {0.0f, -1.0f, nan, infinity}) {
        const auto model = valid_model();
        const auto kernel = valid_kernel();
        require_throws([&] {
            (void)quadratic::prepare_dynamics(model, kernel, invalid_dt);
        }, "Quadratic rough-Heston accepted an invalid time step.");
    }
    for (const float invalid_horizon : {0.0f, -1.0f, nan, infinity}) {
        const auto model = valid_model();
        require_throws([&] {
            (void)quadratic::prepare_dynamics<2U>(
                model,
                invalid_horizon,
                1.0f / 252.0f
            );
        }, "Quadratic rough-Heston accepted an invalid fitting horizon.");
    }

    for (std::size_t factor = 0U; factor < 2U; ++factor) {
        for (const float invalid : {0.0f, -1.0f, nan, infinity}) {
            auto kernel = valid_kernel();
            kernel.nodes[factor] = invalid;
            require_throws([&] {
                (void)quadratic::prepare_dynamics(
                    valid_model(),
                    kernel,
                    1.0f / 252.0f
                );
            }, "Quadratic rough-Heston accepted an invalid kernel node.");

            kernel = valid_kernel();
            kernel.weights[factor] = invalid;
            require_throws([&] {
                (void)quadratic::prepare_dynamics(
                    valid_model(),
                    kernel,
                    1.0f / 252.0f
                );
            }, "Quadratic rough-Heston accepted an invalid kernel weight.");
        }
    }

    auto overflowing_model = valid_model();
    overflowing_model.risk_free_rate = std::numeric_limits<float>::max();
    overflowing_model.dividend_yield = -std::numeric_limits<float>::max();
    require_throws<std::overflow_error>([&] {
        (void)quadratic::prepare_dynamics(
            overflowing_model,
            valid_kernel(),
            2.0f
        );
    }, "Quadratic rough-Heston retained a non-finite prepared drift.");

    auto boundary_model = valid_model();
    boundary_model.risk_free_rate = -0.05f;
    boundary_model.dividend_yield = -0.10f;
    boundary_model.initial_feedback = -0.25f;
    boundary_model.quadratic_shift = -0.50f;
    boundary_model.hurst_exponent = 0.499f;
    const auto prepared = quadratic::prepare_dynamics(
        boundary_model,
        valid_kernel(),
        1.0e-6f
    );
    require(
        std::isfinite(prepared.drift_dt)
            && std::isfinite(prepared.feedback_cell_loading)
            && prepared.feedback_cell_loading > 0.0f,
        "Valid quadratic rough-Heston boundary inputs were not prepared."
    );

    auto invalid_shared_kernel = valid_kernel();
    invalid_shared_kernel.nodes[0] = 0.0f;
    const rough_heston::ModelParameters rough_model{
        1.0f, 0.02f, 0.01f, 0.04f, 1.0f, 0.04f, 0.5f, 0.1f, -0.5f,
    };
    require_throws([&] {
        (void)rough_heston::prepare_dynamics(
            rough_model,
            invalid_shared_kernel,
            1.0f,
            1.0f / 252.0f
        );
    }, "Rough-Heston and quadratic rough-Heston kernel domains diverged.");

    constexpr float horizon = 2.0f;
    constexpr float dt = 1.0f / 504.0f;
    constexpr float minimum_hurst = 0.01f;
    constexpr float maximum_hurst = 0.20f;
    constexpr std::size_t grid_point_count = 257U;
    const auto grid =
        volterra::fit_positive_fractional_kernel_l2_hurst_grid<
            7U,
            grid_point_count
        >(minimum_hurst, maximum_hurst, horizon, dt);
    constexpr std::size_t probe_cells[] = {0U, 1U, 16U, 64U, 128U, 192U, 255U};
    double maximum_interpolation_penalty = 0.0;
    double maximum_interpolated_error = 0.0;
    for (std::size_t cell = 0U;
         cell + 1U < grid_point_count;
         ++cell) {
        const float position = (
            static_cast<float>(cell) + 0.5f
        ) / static_cast<float>(grid_point_count - 1U);
        const float hurst = std::lerp(
            minimum_hurst,
            maximum_hurst,
            position
        );
        maximum_interpolated_error = std::max(
            maximum_interpolated_error,
            volterra::fractional_kernel_relative_l2_error(
                grid.interpolate(hurst),
                hurst,
                horizon,
                dt
            )
        );
    }
    for (const std::size_t cell : probe_cells) {
        const float position = (
            static_cast<float>(cell) + 0.5f
        ) / static_cast<float>(grid_point_count - 1U);
        const float hurst = std::lerp(
            minimum_hurst,
            maximum_hurst,
            position
        );
        const auto interpolated = grid.interpolate(hurst);
        const auto exact = volterra::fit_positive_fractional_kernel_l2<7U>(
            hurst,
            horizon,
            dt
        );
        const double interpolated_error =
            volterra::fractional_kernel_relative_l2_error(
                interpolated,
                hurst,
                horizon,
                dt
            );
        const double exact_error =
            volterra::fractional_kernel_relative_l2_error(
                exact,
                hurst,
                horizon,
                dt
            );
        maximum_interpolation_penalty = std::max(
            maximum_interpolation_penalty,
            interpolated_error - exact_error
        );
    }
    require(
        maximum_interpolated_error < 1.1e-3,
        "Quadratic rough-Heston H-grid kernel error exceeds 0.11%."
    );
    require(
        maximum_interpolation_penalty < 5.0e-6,
        "Quadratic rough-Heston H-grid interpolation penalty is too large."
    );

    for (const float hurst : {0.005f, 0.01f, 0.10f, 0.20f, 0.45f}) {
        const auto kernel_2 =
            volterra::fit_positive_fractional_kernel_l2<2U>(
                hurst, horizon, dt
            );
        const auto kernel_3 =
            volterra::fit_positive_fractional_kernel_l2<3U>(
                hurst, horizon, dt
            );
        const auto kernel_7 =
            volterra::fit_positive_fractional_kernel_l2<7U>(
                hurst, horizon, dt
            );
        const double error_2 = volterra::fractional_kernel_relative_l2_error(
            kernel_2, hurst, horizon, dt
        );
        const double error_3 = volterra::fractional_kernel_relative_l2_error(
            kernel_3, hurst, horizon, dt
        );
        const double error_7 = volterra::fractional_kernel_relative_l2_error(
            kernel_7, hurst, horizon, dt
        );
        std::printf(
            "QUADRATIC_ROUGH_HESTON_KERNEL,H=%.6f,N2=%.9g,N3=%.9g,N7=%.9g\n",
            hurst,
            error_2,
            error_3,
            error_7
        );
        require(
            error_7 < error_3 && error_3 < error_2,
            "Quadratic rough-Heston kernel error did not contract with N."
        );
        require(
            error_7 < 1.1e-3,
            "Quadratic rough-Heston seven-factor kernel exceeds 0.11% L2."
        );
    }
}
