// Schobel--Zhu parameter access and dynamics identity for device preparation.
#pragma once

#include "model/equity/markovian/schobel_zhu/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::schobel_zhu::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.initial_volatility"},
        std::string_view{"model.mean_reversion"},
        std::string_view{"model.long_run_volatility"},
        std::string_view{"model.volatility_of_volatility"},
        std::string_view{"model.correlation"},
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
        case 3U: return model.initial_volatility;
        case 4U: return model.mean_reversion;
        case 5U: return model.long_run_volatility;
        case 6U: return model.volatility_of_volatility;
        case 7U: return model.correlation;
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
        case 3U: model.initial_volatility = value; break;
        case 4U: model.mean_reversion = value; break;
        case 5U: model.long_run_volatility = value; break;
        case 6U: model.volatility_of_volatility = value; break;
        case 7U: model.correlation = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(
        const Model& first,
        const Model& second
    ) {
        return first.risk_free_rate == second.risk_free_rate
            && first.dividend_yield == second.dividend_yield
            && first.initial_volatility == second.initial_volatility
            && first.mean_reversion == second.mean_reversion
            && first.long_run_volatility == second.long_run_volatility
            && first.volatility_of_volatility == second.volatility_of_volatility
            && first.correlation == second.correlation;
    }
};

}  // namespace ai_factory::workbench::model::equity::schobel_zhu::price_gradients
