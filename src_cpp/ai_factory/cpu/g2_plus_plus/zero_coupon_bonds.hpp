#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::g2_plus_plus {
void price_zero_coupon_bond_batch(const cuda::G2PlusPlusZeroCouponBondRow*,std::size_t,cuda::MonteCarloOutput*);
}
