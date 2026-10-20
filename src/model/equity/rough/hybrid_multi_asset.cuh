// The four Gaussian-Volterra equity models share the same correlated FFT scan.
#pragma once

#include "common/equity/absorbing_lamperti.cuh"
#include "common/volterra/fractional_hybrid_kernel.cuh"
#include "common/volterra/fractional_resolvent_hybrid_kernel.cuh"
#include "common/volterra/log_modulated_hybrid_kernel.cuh"
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
        const PreparedModel& p,float value,float variance
    ) {
        return expf(p.log_initial_variance+p.eta*value
                    -p.half_eta_squared*variance);
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& p,float variance,float normal,float& log_spot
    ) {
        log_spot=fmaf(sqrtf(variance)*p.sqrt_time_step,normal,
            log_spot+p.drift_time_step-0.5f*variance*p.time_step);
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
        const PreparedModel& p,float value,float variance
    ) {
        return expf(p.log_initial_variance+p.eta*value
                    -p.half_eta_squared*variance);
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& p,float variance,float normal,float& log_spot
    ) {
        log_spot=fmaf(sqrtf(variance)*p.sqrt_time_step,normal,
            log_spot+p.drift_time_step-0.5f*variance*p.time_step);
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
        const PreparedModel& p,float value,float variance
    ) {
        return expf(p.log_initial_volatility+p.half_eta*value
                    -p.quarter_eta_squared*variance);
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& p,float volatility,float normal,float& log_spot
    ) {
        if (p.one_minus_beta<1.0e-4f) {
            log_spot+=p.drift_time_step
                -0.5f*volatility*volatility*p.time_step
                +volatility*p.sqrt_time_step*normal;
        } else {
            ::ai_factory::workbench::equity::advance_absorbing_lamperti_log_spot(
                p.beta,p.one_minus_beta,p.drift_time_step,p.time_step,
                p.sqrt_time_step,volatility,normal,log_spot);
        }
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
        const PreparedModel& p,float value,float
    ) {
        return p.volatility_level+p.volatility_of_volatility*value;
    }
    __device__ static void advance_spot_cell(
        const PreparedModel& p,float volatility,float normal,float& log_spot
    ) {
        log_spot=fmaf(volatility*p.sqrt_time_step,normal,
            log_spot+p.drift_time_step
                -0.5f*volatility*volatility*p.time_step);
    }
};

}  // namespace ai_factory::workbench::model::equity::rough_multi_asset
