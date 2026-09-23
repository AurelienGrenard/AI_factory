// Vasicek coordinates for compact device sensitivity preparation.
#pragma once

#include "model/fixed_income/vasicek/parameter_domain.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::fixed_income::vasicek::price_gradients {
struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kSupportsMaturityDiagonal = false;
    static constexpr std::array parameter_names{
        std::string_view{"model.mean_reversion"},
        std::string_view{"model.long_term_mean"},
        std::string_view{"model.volatility"},
        std::string_view{"model.initial_state"},
    };
    __host__ __device__ static bool valid(const Model& m) {
        return valid_parameters(m);
    }
    __host__ __device__ static float read(std::uint8_t i, const Model& m) {
        switch (i) {
        case 0U: return m.process.mean_reversion;
        case 1U: return m.process.long_term_mean;
        case 2U: return m.process.volatility;
        case 3U: return m.initial_state;
        }
        return ::nanf("");
    }
    __host__ __device__ static void write(std::uint8_t i, Model& m, float v) {
        switch (i) {
        case 0U: m.process.mean_reversion = v; break;
        case 1U: m.process.long_term_mean = v; break;
        case 2U: m.process.volatility = v; break;
        case 3U: m.initial_state = v; break;
        }
    }
};
}  // namespace ai_factory::workbench::model::fixed_income::vasicek::price_gradients
