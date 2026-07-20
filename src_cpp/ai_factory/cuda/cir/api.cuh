#pragma once
#include "ai_factory/cuda/common/types.cuh"
namespace ai_factory::cuda {
void price_cir_caplet_cuda(const CirCapletRow*,std::size_t,std::size_t,double,MonteCarloOutput*,CudaTiming*);
void price_cir_interest_rate_swap_cuda(const CirSwapRow*,std::size_t,MonteCarloOutput*,CudaTiming*);
void price_cir_zero_coupon_bond_cuda(const CirZeroCouponBondRow*,std::size_t,MonteCarloOutput*,CudaTiming*);
void price_cir_swaption_cuda(const CirSwaptionRow*,std::size_t,std::size_t,double,MonteCarloOutput*,CudaTiming*);
void price_cir_bermudan_swaption_cuda(const CirBermudanSwaptionRow*,std::size_t,std::size_t,double,MonteCarloOutput*,CudaTiming*);
}
