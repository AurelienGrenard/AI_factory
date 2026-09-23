// CIR device adapter for compact sensitivity preparation.
#pragma once

#include "model/fixed_income/cir/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::fixed_income::cir::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kSupportsMaturityDiagonal = false;
    static constexpr std::array parameter_names{
        std::string_view{"model.mean_reversion"},
        std::string_view{"model.long_term_mean"},
        std::string_view{"model.volatility"},
        std::string_view{"model.initial_state"},
    };

    __host__ __device__ static bool valid(const Model& model) {
        return valid_parameters(model);
    }

    __host__ __device__ static float read(
        std::uint8_t index,
        const Model& model
    ) {
        switch (index) {
        case 0U: return model.process.mean_reversion;
        case 1U: return model.process.long_term_mean;
        case 2U: return model.process.volatility;
        case 3U: return model.initial_state;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(
        std::uint8_t index,
        Model& model,
        float value
    ) {
        switch (index) {
        case 0U: model.process.mean_reversion = value; break;
        case 1U: model.process.long_term_mean = value; break;
        case 2U: model.process.volatility = value; break;
        case 3U: model.initial_state = value; break;
        }
    }
};

}  // namespace ai_factory::workbench::model::fixed_income::cir::price_gradients
