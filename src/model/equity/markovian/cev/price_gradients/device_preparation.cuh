// CEV device adapter for parameter access and scenario validation.
#pragma once

#include "model/equity/markovian/cev/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::cev::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kMultiplicativeSpot = false;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.sigma"},
        std::string_view{"model.beta"},
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
        case 3U: return model.sigma;
        case 4U: return model.beta;
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
        case 3U: model.sigma = value; break;
        case 4U: model.beta = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(
        const Model& first,
        const Model& second
    ) {
        return first.risk_free_rate == second.risk_free_rate
            && first.dividend_yield == second.dividend_yield
            && first.sigma == second.sigma
            && first.beta == second.beta;
    }
};

}  // namespace ai_factory::workbench::model::equity::cev::price_gradients
