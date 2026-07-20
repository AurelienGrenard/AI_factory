#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::hull_white {
void price_zero_coupon_bond_batch(const cuda::HullWhiteZeroCouponBondRow*,std::size_t,cuda::MonteCarloOutput*);
}
