#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_heston_european_call_cuda(
    const HestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_heston_digital_call_cuda(
    const HestonRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);

#define AI_FACTORY_DECLARE_HESTON_BARRIER(name) \
    void price_heston_##name##_cuda( \
        const HestonBarrierRow*, std::size_t, std::size_t, std::size_t, \
        MonteCarloOutput*, CudaTiming* \
    )

AI_FACTORY_DECLARE_HESTON_BARRIER(down_and_out_call);
AI_FACTORY_DECLARE_HESTON_BARRIER(down_and_in_call);
AI_FACTORY_DECLARE_HESTON_BARRIER(up_and_out_call);
AI_FACTORY_DECLARE_HESTON_BARRIER(up_and_in_call);

#undef AI_FACTORY_DECLARE_HESTON_BARRIER

void price_heston_autocall_cuda(
    const HestonAutocallRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    AutocallOutput* host_outputs,
    CudaTiming* timing
);

HestonCudaWorkspace* create_heston_workspace(std::size_t row_capacity);

void destroy_heston_workspace(HestonCudaWorkspace* workspace);

void price_heston_cuda_workspace(
    HestonCudaWorkspace* workspace,
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_heston_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_heston_lookback_fixed_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_heston_lookback_fixed_delta_crn_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
);

void price_heston_asian_arithmetic_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_heston_asian_arithmetic_delta_crn_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
);

void price_heston_american_put_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_heston_volatility_swap_cuda(
    const HestonRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void generate_heston_terminal_spots_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_terminal_spots,
    CudaTiming* timing
);

void generate_heston_max_spots_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_max_spots,
    CudaTiming* timing
);

void generate_heston_spot_paths_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    CudaTiming* timing
);

void generate_heston_state_paths_cuda(
    const HestonRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    double* host_variance_paths,
    CudaTiming* timing
);

}  // namespace ai_factory::cuda
