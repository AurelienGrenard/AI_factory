#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_rough_bergomi_european_call_cuda(
    const RoughBergomiRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_rough_bergomi_digital_call_cuda(
    const RoughBergomiRow*, std::size_t, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);

#define AI_FACTORY_DECLARE_ROUGH_BERGOMI_BARRIER(name) \
    void price_rough_bergomi_##name##_cuda( \
        const RoughBergomiBarrierRow*, std::size_t, std::size_t, std::size_t, \
        MonteCarloOutput*, CudaTiming* \
    )

AI_FACTORY_DECLARE_ROUGH_BERGOMI_BARRIER(down_and_out_call);
AI_FACTORY_DECLARE_ROUGH_BERGOMI_BARRIER(down_and_in_call);
AI_FACTORY_DECLARE_ROUGH_BERGOMI_BARRIER(up_and_out_call);
AI_FACTORY_DECLARE_ROUGH_BERGOMI_BARRIER(up_and_in_call);

#undef AI_FACTORY_DECLARE_ROUGH_BERGOMI_BARRIER

void price_rough_bergomi_autocall_cuda(
    const RoughBergomiAutocallRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    AutocallOutput* host_outputs,
    CudaTiming* timing
);

RoughBergomiCudaWorkspace* create_rough_bergomi_workspace(
    std::size_t row_capacity
);

void destroy_rough_bergomi_workspace(RoughBergomiCudaWorkspace* workspace);

void price_rough_bergomi_cuda_workspace(
    RoughBergomiCudaWorkspace* workspace,
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_rough_bergomi_cuda(
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_rough_bergomi_delta_crn_cuda(
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
);

void price_rough_bergomi_asian_arithmetic_cuda(
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void price_rough_bergomi_asian_arithmetic_delta_crn_cuda(
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    PriceDeltaOutput* host_outputs,
    CudaTiming* timing
);

void price_rough_bergomi_volatility_swap_cuda(
    const RoughBergomiRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
);

void generate_rough_bergomi_spot_paths_cuda(
    const RoughBergomiRow* host_row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* host_spot_paths,
    CudaTiming* timing
);

}  // namespace ai_factory::cuda
