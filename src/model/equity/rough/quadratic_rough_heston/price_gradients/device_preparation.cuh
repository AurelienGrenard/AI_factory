// Device preparation maps quadratic_rough_heston parameters into the shared sensitivity graph.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/parameters.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.initial_feedback"},
        std::string_view{"model.quadratic_scale"},
        std::string_view{"model.quadratic_shift"},
        std::string_view{"model.variance_floor"},
        std::string_view{"model.feedback_rate"},
        std::string_view{"model.feedback_volatility"},
        std::string_view{"model.hurst_exponent"}
    };

    __host__ __device__ static bool finite(float value) {
#if defined(__CUDA_ARCH__)
        return ::isfinite(value);
#else
        return std::isfinite(value);
#endif
    }

    __host__ __device__ static bool valid(const Model& m) {
        return finite(m.spot) && m.spot > 0.0f && finite(m.risk_free_rate)
            && finite(m.dividend_yield) && finite(m.initial_feedback)
            && finite(m.quadratic_scale) && m.quadratic_scale > 0.0f
            && finite(m.quadratic_shift) && finite(m.variance_floor)
            && m.variance_floor > 0.0f && finite(m.feedback_rate)
            && m.feedback_rate > 0.0f && finite(m.feedback_volatility)
            && m.feedback_volatility > 0.0f && finite(m.hurst_exponent)
            && m.hurst_exponent > 0.0f && m.hurst_exponent < 0.5f;
    }

    __host__ __device__ static float read(
        std::uint8_t index, const Model& model
    ) {
        switch (index) {
        case 0U: return model.spot;
        case 1U: return model.risk_free_rate;
        case 2U: return model.dividend_yield;
        case 3U: return model.initial_feedback;
        case 4U: return model.quadratic_scale;
        case 5U: return model.quadratic_shift;
        case 6U: return model.variance_floor;
        case 7U: return model.feedback_rate;
        case 8U: return model.feedback_volatility;
        case 9U: return model.hurst_exponent;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index, Model& model, float value
    ) {
        switch (index) {
        case 0U: model.spot = value; break;
        case 1U: model.risk_free_rate = value; break;
        case 2U: model.dividend_yield = value; break;
        case 3U: model.initial_feedback = value; break;
        case 4U: model.quadratic_scale = value; break;
        case 5U: model.quadratic_shift = value; break;
        case 6U: model.variance_floor = value; break;
        case 7U: model.feedback_rate = value; break;
        case 8U: model.feedback_volatility = value; break;
        case 9U: model.hurst_exponent = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(
        const Model& first, const Model& second
    ) {
        return first.risk_free_rate == second.risk_free_rate
            && first.dividend_yield == second.dividend_yield
            && first.initial_feedback == second.initial_feedback
            && first.quadratic_scale == second.quadratic_scale
            && first.quadratic_shift == second.quadratic_shift
            && first.variance_floor == second.variance_floor
            && first.feedback_rate == second.feedback_rate
            && first.feedback_volatility == second.feedback_volatility
            && first.hurst_exponent == second.hurst_exponent;
    }
};

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston::price_gradients
