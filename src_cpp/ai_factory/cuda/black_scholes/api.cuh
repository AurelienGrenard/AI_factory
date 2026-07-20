#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_black_scholes_european_call_cuda(
    const BlackScholesRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_black_scholes_digital_call_cuda(
    const BlackScholesRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);

#define AI_FACTORY_DECLARE_BLACK_SCHOLES_BARRIER(name) \
    void price_black_scholes_##name##_cuda( \
        const BlackScholesBarrierRow*, std::size_t, std::size_t, std::size_t, \
        MonteCarloOutput*, CudaTiming* \
    )

AI_FACTORY_DECLARE_BLACK_SCHOLES_BARRIER(down_and_out_call);
AI_FACTORY_DECLARE_BLACK_SCHOLES_BARRIER(down_and_in_call);
AI_FACTORY_DECLARE_BLACK_SCHOLES_BARRIER(up_and_out_call);
AI_FACTORY_DECLARE_BLACK_SCHOLES_BARRIER(up_and_in_call);

#undef AI_FACTORY_DECLARE_BLACK_SCHOLES_BARRIER

void price_black_scholes_lookback_fixed_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_black_scholes_lookback_fixed_delta_crn_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
);

void price_black_scholes_asian_arithmetic_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_black_scholes_asian_arithmetic_delta_crn_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
);

void price_black_scholes_volatility_swap_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_black_scholes_american_put_cuda(
    const BlackScholesRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_black_scholes_autocall_cuda(
    const BlackScholesAutocallRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    AutocallOutput* host_outputs,
    CudaTiming* timing
);

void generate_black_scholes_spot_paths_cuda(
    const BlackScholesRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    CudaTiming* timing
);

}  // namespace ai_factory::cuda
