// SABR parameter access for row-local finite-difference nodes.
#pragma once

#include "model/equity/markovian/sabr/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::sabr::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    // SABR's dimensional alpha depends on spot: scaling a completed path is invalid.
    static constexpr bool kMultiplicativeSpot = false;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.initial_volatility"},
        std::string_view{"model.volatility_of_volatility"},
        std::string_view{"model.rho"},
        std::string_view{"model.beta"},
    };

    __host__ __device__ static bool valid(const Model& model) {
        return valid_parameters(model);
    }

    __host__ __device__ static float read(std::uint8_t index, const Model& model) {
        switch (index) {
        case 0U: return model.spot;
        case 1U: return model.risk_free_rate;
        case 2U: return model.dividend_yield;
        case 3U: return model.initial_volatility;
        case 4U: return model.volatility_of_volatility;
        case 5U: return model.rho;
        case 6U: return model.beta;
        }
        return ::nanf("");
    }

    __host__ __device__ static void write(std::uint8_t index, Model& model, float value) {
        switch (index) {
        case 0U: model.spot = value; break;
        case 1U: model.risk_free_rate = value; break;
        case 2U: model.dividend_yield = value; break;
        case 3U: model.initial_volatility = value; break;
        case 4U: model.volatility_of_volatility = value; break;
        case 5U: model.rho = value; break;
        case 6U: model.beta = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(const Model& first, const Model& second) {
        return first.risk_free_rate == second.risk_free_rate
            && first.dividend_yield == second.dividend_yield
            && first.initial_volatility == second.initial_volatility
            && first.volatility_of_volatility == second.volatility_of_volatility
            && first.rho == second.rho
            && first.beta == second.beta;
    }
};

}  // namespace ai_factory::workbench::model::equity::sabr::price_gradients
