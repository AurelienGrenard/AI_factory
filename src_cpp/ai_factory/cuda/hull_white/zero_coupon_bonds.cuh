#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cuda {
void price_hull_white_zero_coupon_bond_cuda(const HullWhiteZeroCouponBondRow*,std::size_t,MonteCarloOutput*,CudaTiming*);
}
