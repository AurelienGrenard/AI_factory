// G2++ coordinates for compact device sensitivity preparation.
#pragma once

#include "model/fixed_income/g2_plus_plus/parameter_domain.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients {
struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kSupportsMaturityDiagonal = false;
    static constexpr std::array parameter_names{
        std::string_view{"model.mean_reversion_x"},
        std::string_view{"model.volatility_x"},
        std::string_view{"model.mean_reversion_y"},
        std::string_view{"model.volatility_y"},
        std::string_view{"model.correlation"},
    };
    __host__ __device__ static bool valid(const Model& m) {
        return valid_parameters(m);
    }
    __host__ __device__ static float read(std::uint8_t i, const Model& m) {
        switch (i) {
        case 0U: return m.process.mean_reversion_x;
        case 1U: return m.process.volatility_x;
        case 2U: return m.process.mean_reversion_y;
        case 3U: return m.process.volatility_y;
        case 4U: return m.process.correlation;
        }
        return ::nanf("");
    }
    __host__ __device__ static void write(std::uint8_t i, Model& m, float v) {
        switch (i) {
        case 0U: m.process.mean_reversion_x = v; break;
        case 1U: m.process.volatility_x = v; break;
        case 2U: m.process.mean_reversion_y = v; break;
        case 3U: m.process.volatility_y = v; break;
        case 4U: m.process.correlation = v; break;
        }
    }
};
}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::price_gradients
