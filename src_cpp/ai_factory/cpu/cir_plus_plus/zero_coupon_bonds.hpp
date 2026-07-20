#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::cir_plus_plus {
void price_zero_coupon_bond_batch(const cuda::CirPlusPlusZeroCouponBondRow*,std::size_t,cuda::MonteCarloOutput*);
}
