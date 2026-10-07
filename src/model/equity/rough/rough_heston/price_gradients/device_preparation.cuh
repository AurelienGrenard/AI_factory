// Device preparation maps rough_heston parameters into the shared sensitivity graph.
#pragma once

#include "model/equity/rough/rough_heston/parameters.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::rough_heston::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.initial_variance"},
        std::string_view{"model.mean_reversion"},
        std::string_view{"model.variance_drift"},
        std::string_view{"model.volatility_of_variance"},
        std::string_view{"model.hurst_exponent"},
        std::string_view{"model.rho"}
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
            && finite(m.dividend_yield) && finite(m.initial_variance)
            && m.initial_variance > 0.0f && finite(m.mean_reversion)
            && m.mean_reversion >= 0.0f && finite(m.variance_drift)
            && m.variance_drift >= 0.0f && finite(m.volatility_of_variance)
            && m.volatility_of_variance > 0.0f && finite(m.hurst_exponent)
            && m.hurst_exponent > 0.0f && m.hurst_exponent < 0.5f
            && finite(m.rho) && m.rho >= -1.0f && m.rho <= 1.0f;
    }

    __host__ __device__ static float read(
        std::uint8_t index, const Model& model
    ) {
        switch (index) {
        case 0U: return model.spot;
        case 1U: return model.risk_free_rate;
        case 2U: return model.dividend_yield;
        case 3U: return model.initial_variance;
        case 4U: return model.mean_reversion;
        case 5U: return model.variance_drift;
        case 6U: return model.volatility_of_variance;
        case 7U: return model.hurst_exponent;
        case 8U: return model.rho;
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
        case 3U: model.initial_variance = value; break;
        case 4U: model.mean_reversion = value; break;
        case 5U: model.variance_drift = value; break;
        case 6U: model.volatility_of_variance = value; break;
        case 7U: model.hurst_exponent = value; break;
        case 8U: model.rho = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(
        const Model& first, const Model& second
    ) {
        return first.risk_free_rate == second.risk_free_rate
            && first.dividend_yield == second.dividend_yield
            && first.initial_variance == second.initial_variance
            && first.mean_reversion == second.mean_reversion
            && first.variance_drift == second.variance_drift
            && first.volatility_of_variance == second.volatility_of_variance
            && first.hurst_exponent == second.hurst_exponent
            && first.rho == second.rho;
    }
};

}  // namespace ai_factory::workbench::model::equity::rough_heston::price_gradients
