// Device preparation maps rough_sabr parameters into the shared sensitivity graph.
#pragma once

#include "model/equity/rough/rough_sabr/parameters.hpp"
#include "common/volterra/fractional_hybrid_kernel.cuh"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::rough_sabr::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    using KernelPolicy = volterra::FractionalHybridKernelPolicy;
    using KernelParameters = typename KernelPolicy::Parameters;
    static_assert(sizeof(KernelParameters) == 1U * sizeof(float));
    __host__ __device__ static KernelParameters kernel_parameters(
        const Model& m
    ) {
        return m.hurst_exponent;
    }
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.xi_0"},
        std::string_view{"model.eta"},
        std::string_view{"model.hurst_exponent"},
        std::string_view{"model.rho"},
        std::string_view{"model.beta"}
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
            && finite(m.xi_0) && m.xi_0 > 0.0f && finite(m.eta) && m.eta >= 0.0f && finite(m.hurst_exponent) && m.hurst_exponent > 0.0f && m.hurst_exponent < 0.5f && finite(m.rho) && m.rho >= -1.0f && m.rho <= 1.0f && finite(m.beta) && m.beta >= 0.5f && m.beta <= 1.0f;
    }
    __host__ __device__ static float read(std::uint8_t index, const Model& m) {
        switch (index) {
        case 0U: return m.spot;
        case 1U: return m.risk_free_rate;
        case 2U: return m.dividend_yield;
        case 3U: return m.xi_0;
        case 4U: return m.eta;
        case 5U: return m.hurst_exponent;
        case 6U: return m.rho;
        case 7U: return m.beta;
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
        case 3U: m.xi_0 = value; break;
        case 4U: m.eta = value; break;
        case 5U: m.hurst_exponent = value; break;
        case 6U: m.rho = value; break;
        case 7U: m.beta = value; break;
        }
    }
    __host__ __device__ static bool same_dynamics(
        const Model& a, const Model& b
    ) {
        return a.risk_free_rate == b.risk_free_rate
            && a.dividend_yield == b.dividend_yield
            && a.xi_0 == b.xi_0
            && a.eta == b.eta
            && a.hurst_exponent == b.hurst_exponent
            && a.rho == b.rho
            && a.beta == b.beta;
    }
};

}  // namespace ai_factory::workbench::model::equity::rough_sabr::price_gradients
