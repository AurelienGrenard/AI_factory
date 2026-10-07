// Device preparation maps rough_stein_stein parameters into the shared sensitivity graph.
#pragma once

#include "model/equity/rough/rough_stein_stein/parameters.hpp"
#include "common/volterra/fractional_resolvent_hybrid_kernel.cuh"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::rough_stein_stein::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    using KernelPolicy = volterra::FractionalResolventHybridKernelPolicy;
    using KernelParameters = typename KernelPolicy::Parameters;
    static_assert(sizeof(KernelParameters) == 2U * sizeof(float));
    __host__ __device__ static KernelParameters kernel_parameters(
        const Model& m
    ) {
        return {m.hurst_exponent, m.mean_reversion};
    }
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.volatility_level"},
        std::string_view{"model.mean_reversion"},
        std::string_view{"model.volatility_of_volatility"},
        std::string_view{"model.hurst_exponent"},
        std::string_view{"model.rho"}
    };

    __host__ __device__ static bool finite(float x) {
#if defined(__CUDA_ARCH__)
        return ::isfinite(x);
#else
        return std::isfinite(x);
#endif
    }
    __host__ __device__ static bool valid(const Model& m) {
        return finite(m.spot) && m.spot > 0.0f
            && finite(m.risk_free_rate) && finite(m.dividend_yield)
            && finite(m.volatility_level) && m.volatility_level >= 0.0f && finite(m.mean_reversion) && m.mean_reversion >= 0.0f && finite(m.volatility_of_volatility) && m.volatility_of_volatility > 0.0f && finite(m.hurst_exponent) && m.hurst_exponent > 0.0f && m.hurst_exponent < 0.5f && finite(m.rho) && m.rho >= -1.0f && m.rho <= 1.0f;
    }
    __host__ __device__ static float read(std::uint8_t index, const Model& m) {
        switch (index) {
        case 0U: return m.spot;
        case 1U: return m.risk_free_rate;
        case 2U: return m.dividend_yield;
        case 3U: return m.volatility_level;
        case 4U: return m.mean_reversion;
        case 5U: return m.volatility_of_volatility;
        case 6U: return m.hurst_exponent;
        case 7U: return m.rho;
        }
        return ::nanf("");
    }
    __host__ __device__ static void write(
        std::uint8_t index, Model& m, float value
    ) {
        switch (index) {
        case 0U: m.spot = value; break;
        case 1U: m.risk_free_rate = value; break;
        case 2U: m.dividend_yield = value; break;
        case 3U: m.volatility_level = value; break;
        case 4U: m.mean_reversion = value; break;
        case 5U: m.volatility_of_volatility = value; break;
        case 6U: m.hurst_exponent = value; break;
        case 7U: m.rho = value; break;
        }
    }
    __host__ __device__ static bool same_dynamics(
        const Model& a, const Model& b
    ) {
        return a.risk_free_rate == b.risk_free_rate
            && a.dividend_yield == b.dividend_yield
            && a.volatility_level == b.volatility_level
            && a.mean_reversion == b.mean_reversion
            && a.volatility_of_volatility == b.volatility_of_volatility
            && a.hurst_exponent == b.hurst_exponent
            && a.rho == b.rho;
    }
};

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein::price_gradients
