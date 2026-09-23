// Kou device adapter for parameter access, domains and dynamics identity.
#pragma once

#include "model/equity/markovian/kou/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::kou::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.volatility"},
        std::string_view{"model.jump_intensity"},
        std::string_view{"model.up_probability"},
        std::string_view{"model.positive_jump_rate"},
        std::string_view{"model.negative_jump_rate"},
    };

    __host__ __device__ static bool valid(const Model& model) {
        return valid_parameters(model);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Model& model
    ) {
        switch (index) {
        case 0U: return model.spot;
        case 1U: return model.risk_free_rate;
        case 2U: return model.dividend_yield;
        case 3U: return model.volatility;
        case 4U: return model.jump_intensity;
        case 5U: return model.up_probability;
        case 6U: return model.positive_jump_rate;
        case 7U: return model.negative_jump_rate;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Model& model,
        float value
    ) {
        switch (index) {
        case 0U: model.spot = value; break;
        case 1U: model.risk_free_rate = value; break;
        case 2U: model.dividend_yield = value; break;
        case 3U: model.volatility = value; break;
        case 4U: model.jump_intensity = value; break;
        case 5U: model.up_probability = value; break;
        case 6U: model.positive_jump_rate = value; break;
        case 7U: model.negative_jump_rate = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(
        const Model& first,
        const Model& second
    ) {
        return first.risk_free_rate == second.risk_free_rate
            && first.dividend_yield == second.dividend_yield
            && first.volatility == second.volatility
            && first.jump_intensity == second.jump_intensity
            && first.up_probability == second.up_probability
            && first.positive_jump_rate == second.positive_jump_rate
            && first.negative_jump_rate == second.negative_jump_rate;
    }
};

}  // namespace ai_factory::workbench::model::equity::kou::price_gradients
