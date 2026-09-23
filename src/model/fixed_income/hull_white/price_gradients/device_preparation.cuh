// Hull-White coordinates for compact device sensitivity preparation.
#pragma once

#include "model/fixed_income/hull_white/parameter_domain.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::fixed_income::hull_white::price_gradients {
struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kSupportsMaturityDiagonal = false;
    static constexpr std::array parameter_names{
        std::string_view{"model.mean_reversion"},
        std::string_view{"model.volatility"},
    };
    __host__ __device__ static bool valid(const Model& m) {
        return valid_parameters(m);
    }
    __host__ __device__ static float read(std::uint8_t i, const Model& m) {
        return i == 0U ? m.mean_reversion
            : i == 1U ? m.volatility : ::nanf("");
    }
    __host__ __device__ static void write(std::uint8_t i, Model& m, float v) {
        if (i == 0U) m.mean_reversion = v;
        else if (i == 1U) m.volatility = v;
    }
};
}  // namespace ai_factory::workbench::model::fixed_income::hull_white::price_gradients
