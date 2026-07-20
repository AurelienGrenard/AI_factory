#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_rough_heston_american_put_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);

void generate_rough_heston_spot_paths_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, double*, CudaTiming*
);
void generate_rough_heston_state_paths_cuda(
    const RoughHestonRow*, std::size_t, std::size_t,
    double*, double*, CudaTiming*
);

void price_rough_heston_european_call_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_rough_heston_digital_call_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_rough_heston_asian_arithmetic_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_rough_heston_asian_arithmetic_delta_crn_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t, double,
    PriceDeltaOutput*, CudaTiming*
);
void price_rough_heston_lookback_fixed_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_rough_heston_lookback_fixed_delta_crn_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t, double,
    PriceDeltaOutput*, CudaTiming*
);
void price_rough_heston_volatility_swap_cuda(
    const RoughHestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_rough_heston_autocall_cuda(
    const RoughHestonAutocallRow*, std::size_t, std::size_t, std::size_t,
    AutocallOutput*, CudaTiming*
);

#define AI_FACTORY_DECLARE_ROUGH_HESTON_BARRIER(name) \
    void price_rough_heston_##name##_cuda( \
        const RoughHestonBarrierRow*, std::size_t, std::size_t, std::size_t, \
        MonteCarloOutput*, CudaTiming* \
    )

AI_FACTORY_DECLARE_ROUGH_HESTON_BARRIER(down_and_out_call);
AI_FACTORY_DECLARE_ROUGH_HESTON_BARRIER(down_and_in_call);
AI_FACTORY_DECLARE_ROUGH_HESTON_BARRIER(up_and_out_call);
AI_FACTORY_DECLARE_ROUGH_HESTON_BARRIER(up_and_in_call);
#undef AI_FACTORY_DECLARE_ROUGH_HESTON_BARRIER

}  // namespace ai_factory::cuda
