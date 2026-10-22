// The four Gaussian-Volterra equity models share the same correlated FFT scan.
#pragma once

#include "volterra/fractional_hybrid_kernel.cuh"
#include "volterra/fractional_resolvent_hybrid_kernel.cuh"
#include "volterra/log_modulated_hybrid_kernel.cuh"
#include "model/equity/rough/rough_bergomi/dynamics_impl.cuh"
#include "model/equity/rough/rough_sabr/dynamics_impl.cuh"
#include "model/equity/rough/rough_stein_stein/dynamics_impl.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dynamics_impl.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::equity::rough_multi_asset {

template<typename PathPolicy>
struct HybridPolicy;

template<>
struct HybridPolicy<rough_bergomi::PathPolicy> {
    using Path = rough_bergomi::PathPolicy;
    using Parameters = Path::Parameters;
    using PreparedModel = Path::PreparedModel;
    using KernelPolicy = volterra::FractionalHybridKernelPolicy;
    static constexpr unsigned int kBrownianDimension = 2U;
    static constexpr bool kUsesVolterraVariance = true;
    __device__ static auto kernel_parameters(const Parameters& p) {
        return Path::kernel_parameters(p);
    }
    __device__ static PreparedModel prepare_model(const Parameters& p,float dt) {
        return Path::prepare_model(p,dt);
    }
    __device__ static float initial_volatility(const PreparedModel& p) {
        return p.initial_variance;
    }
    __device__ static float map_volatility(
        const PreparedModel& model, float value, float variance
    ) {
        return rough_bergomi::map_volatility(model, value, variance);
    }
    __device__ static void advance_spot_cell_with_rate(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot, float carry_shift
    ) {
        auto adjusted = model;
        adjusted.drift_time_step += carry_shift;
        rough_bergomi::advance_spot_cell(
            adjusted, volatility, normal, log_spot);
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot
    ) {
        rough_bergomi::advance_spot_cell(model, volatility, normal, log_spot);
    }
};

template<>
struct HybridPolicy<log_modulated_rough_bergomi::PathPolicy> {
    using Path = log_modulated_rough_bergomi::PathPolicy;
    using Parameters = Path::Parameters;
    using PreparedModel = Path::PreparedModel;
    using KernelPolicy = volterra::LogModulatedHybridKernelPolicy;
    static constexpr unsigned int kBrownianDimension = 2U;
    static constexpr bool kUsesVolterraVariance = true;
    __device__ static auto kernel_parameters(const Parameters& p) {
        return Path::kernel_parameters(p);
    }
    __device__ static PreparedModel prepare_model(const Parameters& p,float dt) {
        return Path::prepare_model(p,dt);
    }
    __device__ static float initial_volatility(const PreparedModel& p) {
        return p.initial_variance;
    }
    __device__ static float map_volatility(
        const PreparedModel& model, float value, float variance
    ) {
        return log_modulated_rough_bergomi::map_volatility(
            model, value, variance);
    }
    __device__ static void advance_spot_cell_with_rate(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot, float carry_shift
    ) {
        auto adjusted = model;
        adjusted.drift_time_step += carry_shift;
        log_modulated_rough_bergomi::advance_spot_cell(
            adjusted, volatility, normal, log_spot);
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot
    ) {
        log_modulated_rough_bergomi::advance_spot_cell(
            model, volatility, normal, log_spot);
    }
};

template<>
struct HybridPolicy<rough_sabr::PathPolicy> {
    using Path = rough_sabr::PathPolicy;
    using Parameters = Path::Parameters;
    using PreparedModel = Path::PreparedModel;
    using KernelPolicy = volterra::FractionalHybridKernelPolicy;
    static constexpr unsigned int kBrownianDimension = 2U;
    static constexpr bool kUsesVolterraVariance = true;
    __device__ static auto kernel_parameters(const Parameters& p) {
        return Path::kernel_parameters(p);
    }
    __device__ static PreparedModel prepare_model(const Parameters& p,float dt) {
        return Path::prepare_model(p,dt);
    }
    __device__ static float initial_volatility(const PreparedModel& p) {
        return p.initial_volatility;
    }
    __device__ static float map_volatility(
        const PreparedModel& model, float value, float variance
    ) {
        return rough_sabr::map_volatility(model, value, variance);
    }
    __device__ static void advance_spot_cell_with_rate(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot, float carry_shift
    ) {
        auto adjusted = model;
        adjusted.drift_time_step += carry_shift;
        rough_sabr::advance_spot_cell(
            adjusted, volatility, normal, log_spot);
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot
    ) {
        rough_sabr::advance_spot_cell(model, volatility, normal, log_spot);
    }
};

template<>
struct HybridPolicy<rough_stein_stein::PathPolicy> {
    using Path = rough_stein_stein::PathPolicy;
    using Parameters = Path::Parameters;
    using PreparedModel = Path::PreparedModel;
    using KernelPolicy = volterra::FractionalResolventHybridKernelPolicy;
    static constexpr unsigned int kBrownianDimension = 2U;
    static constexpr bool kUsesVolterraVariance = false;
    __device__ static auto kernel_parameters(const Parameters& p) {
        return Path::kernel_parameters(p);
    }
    __device__ static PreparedModel prepare_model(const Parameters& p,float dt) {
        return Path::prepare_model(p,dt);
    }
    __device__ static float initial_volatility(const PreparedModel& p) {
        return p.volatility_level;
    }
    __device__ static float map_volatility(
        const PreparedModel& model, float value, float variance
    ) {
        return rough_stein_stein::map_volatility(model, value, variance);
    }
    __device__ static void advance_spot_cell_with_rate(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot, float carry_shift
    ) {
        auto adjusted = model;
        adjusted.drift_time_step += carry_shift;
        rough_stein_stein::advance_spot_cell(
            adjusted, volatility, normal, log_spot);
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& model, float volatility,
        float normal, float& log_spot
    ) {
        rough_stein_stein::advance_spot_cell(
            model, volatility, normal, log_spot);
    }
};

}  // namespace ai_factory::workbench::model::equity::rough_multi_asset
