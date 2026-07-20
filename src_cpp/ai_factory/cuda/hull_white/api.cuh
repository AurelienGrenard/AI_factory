#pragma once

#include "ai_factory/cuda/common/types.cuh"

namespace ai_factory::cuda {

void price_hull_white_interest_rate_swap_cuda(
    const HullWhiteSwapRow*, std::size_t, MonteCarloOutput*, CudaTiming*
);
void price_hull_white_zero_coupon_bond_cuda(
    const HullWhiteZeroCouponBondRow*,std::size_t,MonteCarloOutput*,CudaTiming*
);

void price_hull_white_swaption_cuda(
    const HullWhiteSwaptionRow*, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);
void price_hull_white_caplet_cuda(
    const HullWhiteCapletRow*, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);

void price_hull_white_bermudan_swaption_cuda(
    const HullWhiteBermudanSwaptionRow*, std::size_t, std::size_t,
    MonteCarloOutput*, CudaTiming*
);

}  // namespace ai_factory::cuda
