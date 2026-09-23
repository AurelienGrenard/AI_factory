// Bates device adapter for parameter access, domains and dynamics identity.
#pragma once

#include "model/equity/markovian/bates/parameter_domain.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace ai_factory::workbench::model::equity::bates::price_gradients {

struct DevicePreparation {
    using Model = ModelParameters;
    static constexpr bool kMultiplicativeSpot = true;
    static constexpr bool kSupportsMaturitySensitivity = true;
    static constexpr bool kSupportsMaturityDiagonal = true;
    static constexpr std::array parameter_names{
        std::string_view{"model.spot"},
        std::string_view{"model.risk_free_rate"},
        std::string_view{"model.dividend_yield"},
        std::string_view{"model.initial_variance"},
        std::string_view{"model.kappa"},
        std::string_view{"model.theta"},
        std::string_view{"model.gamma"},
        std::string_view{"model.rho"},
        std::string_view{"model.jump_intensity"},
        std::string_view{"model.jump_log_mean"},
        std::string_view{"model.jump_log_volatility"},
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
        case 3U: return model.initial_variance;
        case 4U: return model.kappa;
        case 5U: return model.theta;
        case 6U: return model.gamma;
        case 7U: return model.rho;
        case 8U: return model.jump_intensity;
        case 9U: return model.jump_log_mean;
        case 10U: return model.jump_log_volatility;
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
        case 3U: model.initial_variance = value; break;
        case 4U: model.kappa = value; break;
        case 5U: model.theta = value; break;
        case 6U: model.gamma = value; break;
        case 7U: model.rho = value; break;
        case 8U: model.jump_intensity = value; break;
        case 9U: model.jump_log_mean = value; break;
        case 10U: model.jump_log_volatility = value; break;
        }
    }

    __host__ __device__ static bool same_dynamics(
        const Model& first,
        const Model& second
    ) {
        return first.risk_free_rate == second.risk_free_rate
            && first.dividend_yield == second.dividend_yield
            && first.initial_variance == second.initial_variance
            && first.kappa == second.kappa
            && first.theta == second.theta
            && first.gamma == second.gamma
            && first.rho == second.rho
            && first.jump_intensity == second.jump_intensity
            && first.jump_log_mean == second.jump_log_mean
            && first.jump_log_volatility == second.jump_log_volatility;
    }
};

}  // namespace ai_factory::workbench::model::equity::bates::price_gradients
