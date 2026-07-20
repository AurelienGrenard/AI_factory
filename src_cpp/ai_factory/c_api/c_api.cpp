#include "ai_factory/cuda/common/api.cuh"
#include "ai_factory/cuda/black_scholes/api.cuh"
#include "ai_factory/cuda/heston/api.cuh"
#include "ai_factory/cuda/rough_bergomi/api.cuh"
#include "ai_factory/cuda/rough_heston/api.cuh"
#include "ai_factory/cuda/hull_white/api.cuh"
#include "ai_factory/cuda/cir/api.cuh"
#include "ai_factory/cuda/cir_plus_plus/api.cuh"
#include "ai_factory/cuda/g2_plus_plus/api.cuh"
#include "ai_factory/cuda/black_76/api.cuh"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/black_scholes/american_puts.hpp"
#include "ai_factory/cpu/black_scholes/european_calls.hpp"
#include "ai_factory/cpu/black_scholes/digital_calls.hpp"
#include "ai_factory/cpu/black_scholes/autocalls.hpp"
#include "ai_factory/cpu/black_scholes/asian_arithmetic_calls.hpp"
#include "ai_factory/cpu/black_scholes/common.hpp"
#include "ai_factory/cpu/black_scholes/lookback_fixed_calls.hpp"
#include "ai_factory/cpu/black_scholes/volatility_swaps.hpp"
#include "ai_factory/cpu/black_scholes/down_and_out_calls.hpp"
#include "ai_factory/cpu/black_scholes/down_and_in_calls.hpp"
#include "ai_factory/cpu/black_scholes/up_and_out_calls.hpp"
#include "ai_factory/cpu/black_scholes/up_and_in_calls.hpp"
#include "ai_factory/cpu/heston/american_puts.hpp"
#include "ai_factory/cpu/heston/european_calls.hpp"
#include "ai_factory/cpu/heston/digital_calls.hpp"
#include "ai_factory/cpu/heston/autocalls.hpp"
#include "ai_factory/cpu/heston/asian_arithmetic_calls.hpp"
#include "ai_factory/cpu/heston/common.hpp"
#include "ai_factory/cpu/heston/lookback_fixed_calls.hpp"
#include "ai_factory/cpu/heston/volatility_swaps.hpp"
#include "ai_factory/cpu/heston/down_and_out_calls.hpp"
#include "ai_factory/cpu/heston/down_and_in_calls.hpp"
#include "ai_factory/cpu/heston/up_and_out_calls.hpp"
#include "ai_factory/cpu/heston/up_and_in_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/asian_arithmetic_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/european_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/digital_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/autocalls.hpp"
#include "ai_factory/cpu/rough_bergomi/common.hpp"
#include "ai_factory/cpu/rough_bergomi/lookback_fixed_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/volatility_swaps.hpp"
#include "ai_factory/cpu/rough_bergomi/down_and_out_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/down_and_in_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/up_and_out_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/up_and_in_calls.hpp"
#include "ai_factory/cpu/rough_heston/american_puts.hpp"
#include "ai_factory/cpu/rough_heston/asian_arithmetic_calls.hpp"
#include "ai_factory/cpu/rough_heston/european_calls.hpp"
#include "ai_factory/cpu/rough_heston/digital_calls.hpp"
#include "ai_factory/cpu/rough_heston/autocalls.hpp"
#include "ai_factory/cpu/rough_heston/common.hpp"
#include "ai_factory/cpu/rough_heston/lookback_fixed_calls.hpp"
#include "ai_factory/cpu/rough_heston/volatility_swaps.hpp"
#include "ai_factory/cpu/rough_heston/down_and_out_calls.hpp"
#include "ai_factory/cpu/rough_heston/down_and_in_calls.hpp"
#include "ai_factory/cpu/rough_heston/up_and_out_calls.hpp"
#include "ai_factory/cpu/rough_heston/up_and_in_calls.hpp"
#include "ai_factory/cpu/hull_white/interest_rate_swaps.hpp"
#include "ai_factory/cpu/hull_white/zero_coupon_bonds.hpp"
#include "ai_factory/cpu/hull_white/swaptions.hpp"
#include "ai_factory/cpu/hull_white/bermudan_swaptions.hpp"
#include "ai_factory/cpu/hull_white/caplets.hpp"
#include "ai_factory/cpu/cir/interest_rate_swaps.hpp"
#include "ai_factory/cpu/cir/zero_coupon_bonds.hpp"
#include "ai_factory/cpu/cir/swaptions.hpp"
#include "ai_factory/cpu/cir/bermudan_swaptions.hpp"
#include "ai_factory/cpu/cir/caplets.hpp"
#include "ai_factory/cpu/cir_plus_plus/interest_rate_swaps.hpp"
#include "ai_factory/cpu/cir_plus_plus/zero_coupon_bonds.hpp"
#include "ai_factory/cpu/cir_plus_plus/swaptions.hpp"
#include "ai_factory/cpu/cir_plus_plus/bermudan_swaptions.hpp"
#include "ai_factory/cpu/cir_plus_plus/caplets.hpp"
#include "ai_factory/cpu/g2_plus_plus/interest_rate_swaps.hpp"
#include "ai_factory/cpu/g2_plus_plus/zero_coupon_bonds.hpp"
#include "ai_factory/cpu/g2_plus_plus/swaptions.hpp"
#include "ai_factory/cpu/g2_plus_plus/bermudan_swaptions.hpp"
#include "ai_factory/cpu/g2_plus_plus/caplets.hpp"
#include "ai_factory/cpu/black_76/caplets.hpp"
#include "ai_factory/cpu/black_76/swaptions.hpp"

#include <algorithm>
#include <exception>
#include <cstddef>
#include <string>

namespace {

thread_local std::string last_error;

template <typename Function>
int call_with_error_boundary(Function&& function) {
    try {
        function();
        last_error.clear();
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    } catch (...) {
        last_error = "Unknown C++ exception.";
        return 1;
    }
}

}  // namespace

extern "C" {

int ai_factory_price_black_76_caplet_cpu_batch(const ai_factory::cuda::Black76CapletRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::black_76::price_caplet_batch(r,n,o);});}
int ai_factory_price_black_76_caplet_gpu_batch(const ai_factory::cuda::Black76CapletRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_black_76_caplet_cuda(r,n,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_black_76_swaption_cpu_batch(const ai_factory::cuda::Black76SwaptionRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::black_76::price_swaption_batch(r,n,o);});}
int ai_factory_price_black_76_swaption_gpu_batch(const ai_factory::cuda::Black76SwaptionRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_black_76_swaption_cuda(r,n,o,&t);if(s)*s=t.total_ms/1000.0;});}

int ai_factory_price_hull_white_caplet_cpu_batch(const ai_factory::cuda::HullWhiteCapletRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::hull_white::price_caplet_batch(r,n,p,o);});}
int ai_factory_price_hull_white_caplet_gpu_batch(const ai_factory::cuda::HullWhiteCapletRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_hull_white_caplet_cuda(r,n,p,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_cir_caplet_cpu_batch(const ai_factory::cuda::CirCapletRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::cir::price_caplet_batch(r,n,p,dt,o);});}
int ai_factory_price_cir_caplet_gpu_batch(const ai_factory::cuda::CirCapletRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_cir_caplet_cuda(r,n,p,dt,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_cir_plus_plus_caplet_cpu_batch(const ai_factory::cuda::CirPlusPlusCapletRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::cir_plus_plus::price_caplet_batch(r,n,p,dt,o);});}
int ai_factory_price_cir_plus_plus_caplet_gpu_batch(const ai_factory::cuda::CirPlusPlusCapletRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_cir_plus_plus_caplet_cuda(r,n,p,dt,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_g2_plus_plus_caplet_cpu_batch(const ai_factory::cuda::G2PlusPlusCapletRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::g2_plus_plus::price_caplet_batch(r,n,p,o);});}
int ai_factory_price_g2_plus_plus_caplet_gpu_batch(const ai_factory::cuda::G2PlusPlusCapletRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_g2_plus_plus_caplet_cuda(r,n,p,o,&t);if(s)*s=t.total_ms/1000.0;});}

int ai_factory_price_hull_white_interest_rate_swap_cpu_batch(const ai_factory::cuda::HullWhiteSwapRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::hull_white::price_interest_rate_swap_batch(rows,count,outputs);});}
int ai_factory_price_hull_white_interest_rate_swap_gpu_batch(const ai_factory::cuda::HullWhiteSwapRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_hull_white_interest_rate_swap_cuda(rows,count,outputs,&timing);if(seconds)*seconds=timing.total_ms/1000.0;});}
int ai_factory_price_hull_white_zero_coupon_bond_cpu_batch(const ai_factory::cuda::HullWhiteZeroCouponBondRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::hull_white::price_zero_coupon_bond_batch(rows,count,outputs);});}
int ai_factory_price_hull_white_zero_coupon_bond_gpu_batch(const ai_factory::cuda::HullWhiteZeroCouponBondRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_hull_white_zero_coupon_bond_cuda(rows,count,outputs,&timing);if(seconds)*seconds=timing.total_ms/1000.0;});}
int ai_factory_price_hull_white_swaption_cpu_batch(const ai_factory::cuda::HullWhiteSwaptionRow* rows,std::size_t count,std::size_t paths,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::hull_white::price_swaption_batch(rows,count,paths,outputs);});}
int ai_factory_price_hull_white_swaption_gpu_batch(const ai_factory::cuda::HullWhiteSwaptionRow* rows,std::size_t count,std::size_t paths,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_hull_white_swaption_cuda(rows,count,paths,outputs,&timing);if(seconds)*seconds=timing.total_ms/1000.0;});}
int ai_factory_price_cir_interest_rate_swap_cpu_batch(const ai_factory::cuda::CirSwapRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::cir::price_interest_rate_swap_batch(rows,count,outputs);});}
int ai_factory_price_cir_interest_rate_swap_gpu_batch(const ai_factory::cuda::CirSwapRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_cir_interest_rate_swap_cuda(rows,count,outputs,&timing);if(seconds)*seconds=timing.total_ms/1000.0;});}
int ai_factory_price_cir_zero_coupon_bond_cpu_batch(const ai_factory::cuda::CirZeroCouponBondRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::cir::price_zero_coupon_bond_batch(rows,count,outputs);});}
int ai_factory_price_cir_zero_coupon_bond_gpu_batch(const ai_factory::cuda::CirZeroCouponBondRow* rows,std::size_t count,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_cir_zero_coupon_bond_cuda(rows,count,outputs,&timing);if(seconds)*seconds=timing.total_ms/1000.0;});}
int ai_factory_price_cir_swaption_cpu_batch(const ai_factory::cuda::CirSwaptionRow* rows,std::size_t count,std::size_t paths,double target_dt,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::cir::price_swaption_batch(rows,count,paths,target_dt,outputs);});}
int ai_factory_price_cir_swaption_gpu_batch(const ai_factory::cuda::CirSwaptionRow* rows,std::size_t count,std::size_t paths,double target_dt,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_cir_swaption_cuda(rows,count,paths,target_dt,outputs,&timing);if(seconds)*seconds=timing.total_ms/1000.0;});}
int ai_factory_price_hull_white_bermudan_swaption_cpu_batch(const ai_factory::cuda::HullWhiteBermudanSwaptionRow* rows,std::size_t count,std::size_t paths,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::hull_white::price_bermudan_swaption_batch(rows,count,paths,outputs);});}
int ai_factory_price_hull_white_bermudan_swaption_gpu_batch(const ai_factory::cuda::HullWhiteBermudanSwaptionRow* rows,std::size_t count,std::size_t paths,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_hull_white_bermudan_swaption_cuda(rows,count,paths,outputs,&timing);if(seconds)*seconds=timing.simulation_ms/1000.0;});}
int ai_factory_price_cir_bermudan_swaption_cpu_batch(const ai_factory::cuda::CirBermudanSwaptionRow* rows,std::size_t count,std::size_t paths,double target_dt,ai_factory::cuda::MonteCarloOutput* outputs){return call_with_error_boundary([&]{ai_factory::cpu::cir::price_bermudan_swaption_batch(rows,count,paths,target_dt,outputs);});}
int ai_factory_price_cir_bermudan_swaption_gpu_batch(const ai_factory::cuda::CirBermudanSwaptionRow* rows,std::size_t count,std::size_t paths,double target_dt,ai_factory::cuda::MonteCarloOutput* outputs,double* seconds){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_cir_bermudan_swaption_cuda(rows,count,paths,target_dt,outputs,&timing);if(seconds)*seconds=timing.simulation_ms/1000.0;});}

int ai_factory_price_cir_plus_plus_interest_rate_swap_cpu_batch(const ai_factory::cuda::CirPlusPlusSwapRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::cir_plus_plus::price_interest_rate_swap_batch(r,n,o);});}
int ai_factory_price_cir_plus_plus_interest_rate_swap_gpu_batch(const ai_factory::cuda::CirPlusPlusSwapRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_cir_plus_plus_interest_rate_swap_cuda(r,n,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_cir_plus_plus_zero_coupon_bond_cpu_batch(const ai_factory::cuda::CirPlusPlusZeroCouponBondRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::cir_plus_plus::price_zero_coupon_bond_batch(r,n,o);});}
int ai_factory_price_cir_plus_plus_zero_coupon_bond_gpu_batch(const ai_factory::cuda::CirPlusPlusZeroCouponBondRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_cir_plus_plus_zero_coupon_bond_cuda(r,n,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_cir_plus_plus_swaption_cpu_batch(const ai_factory::cuda::CirPlusPlusSwaptionRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::cir_plus_plus::price_swaption_batch(r,n,p,dt,o);});}
int ai_factory_price_cir_plus_plus_swaption_gpu_batch(const ai_factory::cuda::CirPlusPlusSwaptionRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_cir_plus_plus_swaption_cuda(r,n,p,dt,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_cir_plus_plus_bermudan_swaption_cpu_batch(const ai_factory::cuda::CirPlusPlusBermudanSwaptionRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::cir_plus_plus::price_bermudan_swaption_batch(r,n,p,dt,o);});}
int ai_factory_price_cir_plus_plus_bermudan_swaption_gpu_batch(const ai_factory::cuda::CirPlusPlusBermudanSwaptionRow*r,std::size_t n,std::size_t p,double dt,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_cir_plus_plus_bermudan_swaption_cuda(r,n,p,dt,o,&t);if(s)*s=t.total_ms/1000.0;});}

int ai_factory_price_g2_plus_plus_interest_rate_swap_cpu_batch(const ai_factory::cuda::G2PlusPlusSwapRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::g2_plus_plus::price_interest_rate_swap_batch(r,n,o);});}
int ai_factory_price_g2_plus_plus_interest_rate_swap_gpu_batch(const ai_factory::cuda::G2PlusPlusSwapRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_g2_plus_plus_interest_rate_swap_cuda(r,n,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_g2_plus_plus_zero_coupon_bond_cpu_batch(const ai_factory::cuda::G2PlusPlusZeroCouponBondRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::g2_plus_plus::price_zero_coupon_bond_batch(r,n,o);});}
int ai_factory_price_g2_plus_plus_zero_coupon_bond_gpu_batch(const ai_factory::cuda::G2PlusPlusZeroCouponBondRow*r,std::size_t n,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_g2_plus_plus_zero_coupon_bond_cuda(r,n,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_g2_plus_plus_swaption_cpu_batch(const ai_factory::cuda::G2PlusPlusSwaptionRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::g2_plus_plus::price_swaption_batch(r,n,p,o);});}
int ai_factory_price_g2_plus_plus_swaption_gpu_batch(const ai_factory::cuda::G2PlusPlusSwaptionRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_g2_plus_plus_swaption_cuda(r,n,p,o,&t);if(s)*s=t.total_ms/1000.0;});}
int ai_factory_price_g2_plus_plus_bermudan_swaption_cpu_batch(const ai_factory::cuda::G2PlusPlusBermudanSwaptionRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o){return call_with_error_boundary([&]{ai_factory::cpu::g2_plus_plus::price_bermudan_swaption_batch(r,n,p,o);});}
int ai_factory_price_g2_plus_plus_bermudan_swaption_gpu_batch(const ai_factory::cuda::G2PlusPlusBermudanSwaptionRow*r,std::size_t n,std::size_t p,ai_factory::cuda::MonteCarloOutput*o,double*s){return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming t{};ai_factory::cuda::price_g2_plus_plus_bermudan_swaption_cuda(r,n,p,o,&t);if(s)*s=t.total_ms/1000.0;});}

#define AI_FACTORY_DEFINE_BARRIER_C_API(model, row_type, family, singular) \
int ai_factory_price_##model##_##family##_cpu_batch( \
    const ai_factory::cuda::row_type* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, \
    ai_factory::cuda::MonteCarloOutput* outputs \
) { \
    return call_with_error_boundary([&] { \
        ai_factory::cpu::model::price_##singular##_batch( \
            rows, row_count, num_paths, num_steps, outputs \
        ); \
    }); \
} \
int ai_factory_price_##model##_##family##_gpu_batch( \
    const ai_factory::cuda::row_type* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, \
    ai_factory::cuda::MonteCarloOutput* outputs, double* kernel_seconds \
) { \
    return call_with_error_boundary([&] { \
        ai_factory::cuda::CudaTiming timing{}; \
        ai_factory::cuda::price_##model##_##singular##_cuda( \
            rows, row_count, num_paths, num_steps, outputs, &timing \
        ); \
        if (kernel_seconds != nullptr) { \
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0; \
        } \
    }); \
}

AI_FACTORY_DEFINE_BARRIER_C_API(black_scholes, BlackScholesBarrierRow, down_and_out_calls, down_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(black_scholes, BlackScholesBarrierRow, down_and_in_calls, down_and_in_call)
AI_FACTORY_DEFINE_BARRIER_C_API(black_scholes, BlackScholesBarrierRow, up_and_out_calls, up_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(black_scholes, BlackScholesBarrierRow, up_and_in_calls, up_and_in_call)
AI_FACTORY_DEFINE_BARRIER_C_API(heston, HestonBarrierRow, down_and_out_calls, down_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(heston, HestonBarrierRow, down_and_in_calls, down_and_in_call)
AI_FACTORY_DEFINE_BARRIER_C_API(heston, HestonBarrierRow, up_and_out_calls, up_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(heston, HestonBarrierRow, up_and_in_calls, up_and_in_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_bergomi, RoughBergomiBarrierRow, down_and_out_calls, down_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_bergomi, RoughBergomiBarrierRow, down_and_in_calls, down_and_in_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_bergomi, RoughBergomiBarrierRow, up_and_out_calls, up_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_bergomi, RoughBergomiBarrierRow, up_and_in_calls, up_and_in_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_heston, RoughHestonBarrierRow, down_and_out_calls, down_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_heston, RoughHestonBarrierRow, down_and_in_calls, down_and_in_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_heston, RoughHestonBarrierRow, up_and_out_calls, up_and_out_call)
AI_FACTORY_DEFINE_BARRIER_C_API(rough_heston, RoughHestonBarrierRow, up_and_in_calls, up_and_in_call)

#undef AI_FACTORY_DEFINE_BARRIER_C_API

int ai_factory_price_black_scholes_autocall_cpu_batch(
    const ai_factory::cuda::BlackScholesAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::black_scholes::price_autocall_batch(
            rows, row_count, num_paths, num_steps, outputs
        );
    });
}

int ai_factory_price_black_scholes_autocall_gpu_batch(
    const ai_factory::cuda::BlackScholesAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_black_scholes_autocall_cuda(
            rows, row_count, num_paths, num_steps, outputs, &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_autocall_cpu_batch(
    const ai_factory::cuda::HestonAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_autocall_batch(
            rows, row_count, num_paths, num_steps, outputs
        );
    });
}

int ai_factory_price_heston_autocall_gpu_batch(
    const ai_factory::cuda::HestonAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_autocall_cuda(
            rows, row_count, num_paths, num_steps, outputs, &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_rough_bergomi_autocall_cpu_batch(
    const ai_factory::cuda::RoughBergomiAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_autocall_batch(
            rows, row_count, num_paths, num_steps, outputs
        );
    });
}

int ai_factory_price_rough_bergomi_autocall_gpu_batch(
    const ai_factory::cuda::RoughBergomiAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_rough_bergomi_autocall_cuda(
            rows, row_count, num_paths, num_steps, outputs, &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

const char* ai_factory_cuda_last_error() {
    return last_error.c_str();
}

int ai_factory_cuda_warmup() {
    return call_with_error_boundary([] { ai_factory::cuda::warmup_cuda(); });
}

void* ai_factory_create_heston_workspace(std::size_t row_capacity) {
    try {
        last_error.clear();
        return ai_factory::cuda::create_heston_workspace(row_capacity);
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown C++ exception.";
        return nullptr;
    }
}

void* ai_factory_create_rough_bergomi_workspace(std::size_t row_capacity) {
    try {
        last_error.clear();
        return ai_factory::cuda::create_rough_bergomi_workspace(row_capacity);
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    } catch (...) {
        last_error = "Unknown C++ exception.";
        return nullptr;
    }
}

void ai_factory_destroy_heston_workspace(void* workspace) {
    ai_factory::cuda::destroy_heston_workspace(
        static_cast<ai_factory::cuda::HestonCudaWorkspace*>(workspace)
    );
}

void ai_factory_destroy_rough_bergomi_workspace(void* workspace) {
    ai_factory::cuda::destroy_rough_bergomi_workspace(
        static_cast<ai_factory::cuda::RoughBergomiCudaWorkspace*>(workspace)
    );
}

int ai_factory_price_heston_workspace(
    void* workspace,
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_cuda_workspace(
            static_cast<ai_factory::cuda::HestonCudaWorkspace*>(workspace),
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_rough_bergomi_workspace(
    void* workspace,
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_rough_bergomi_cuda_workspace(
            static_cast<ai_factory::cuda::RoughBergomiCudaWorkspace*>(workspace),
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_rough_bergomi_lookback_fixed_cpu(
    const ai_factory::cuda::RoughBergomiRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_lookback_fixed_call(
            *row,
            num_paths,
            num_steps,
            *output
        );
    });
}

int ai_factory_price_rough_bergomi_lookback_fixed_cpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_lookback_fixed_call_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_rough_bergomi_lookback_fixed_delta_crn_cpu(
    const ai_factory::cuda::RoughBergomiRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_lookback_fixed_call_delta_crn(
            *row,
            num_paths,
            num_steps,
            relative_bump,
            *output
        );
    });
}

int ai_factory_price_rough_bergomi_lookback_fixed_delta_crn_cpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_lookback_fixed_call_delta_crn_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs
        );
    });
}

int ai_factory_price_rough_bergomi_lookback_fixed_gpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_rough_bergomi_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_generate_heston_terminal_spots(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* terminal_spots,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::generate_heston_terminal_spots_cuda(
            row,
            num_paths,
            num_steps,
            terminal_spots,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_generate_heston_max_spots(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* max_spots,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::generate_heston_max_spots_cuda(
            row,
            num_paths,
            num_steps,
            max_spots,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_generate_heston_spot_paths(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::generate_heston_spot_paths_cuda(
            row,
            num_paths,
            num_steps,
            spot_paths,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_generate_heston_state_paths(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths,
    double* variance_paths,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::generate_heston_state_paths_cuda(
            row, num_paths, num_steps, spot_paths, variance_paths, &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_cpu_generate_rough_bergomi_spot_paths(
    const ai_factory::cuda::RoughBergomiRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths
) {
    return call_with_error_boundary([&] {
        const ai_factory::simulation::RoughBergomiModel model{
            row->spot,
            row->risk_free_rate,
            row->dividend_yield,
            row->forward_variance,
            row->eta,
            row->alpha,
            row->rho,
        };
        const ai_factory::simulation::TimeGrid time_grid{
            row->maturity,
            num_steps,
        };
        const ai_factory::simulation::SimulationConfig simulation{
            row->seed,
            num_paths,
            ai_factory::simulation::kPhilox4x32_10BoxMuller,
        };
        const auto paths =
            ai_factory::simulation::generate_rough_bergomi_spot_paths(
                model,
                time_grid,
                simulation
            );
        std::copy(paths.begin(), paths.end(), spot_paths);
    });
}

int ai_factory_generate_rough_bergomi_spot_paths(
    const ai_factory::cuda::RoughBergomiRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::generate_rough_bergomi_spot_paths_cuda(
            row,
            num_paths,
            num_steps,
            spot_paths,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_rough_bergomi_lookback_fixed_delta_crn_gpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_rough_bergomi_delta_crn_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_rough_bergomi_asian_arithmetic_cpu(
    const ai_factory::cuda::RoughBergomiRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_asian_arithmetic_call(
            *row,
            num_paths,
            num_steps,
            *output
        );
    });
}

int ai_factory_price_rough_bergomi_asian_arithmetic_cpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_asian_arithmetic_call_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_rough_bergomi_asian_arithmetic_delta_crn_cpu(
    const ai_factory::cuda::RoughBergomiRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_asian_arithmetic_call_delta_crn(
            *row,
            num_paths,
            num_steps,
            relative_bump,
            *output
        );
    });
}

int ai_factory_price_rough_bergomi_asian_arithmetic_delta_crn_cpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_asian_arithmetic_call_delta_crn_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs
        );
    });
}

int ai_factory_price_rough_bergomi_asian_arithmetic_gpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_rough_bergomi_asian_arithmetic_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_rough_bergomi_asian_arithmetic_delta_crn_gpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_rough_bergomi_asian_arithmetic_delta_crn_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_rough_bergomi_volatility_swap_cpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::rough_bergomi::price_volatility_swap_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_rough_bergomi_volatility_swap_gpu_batch(
    const ai_factory::cuda::RoughBergomiRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_rough_bergomi_volatility_swap_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_asian_arithmetic_cpu(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_asian_arithmetic_call(
            *row,
            num_paths,
            num_steps,
            *output
        );
    });
}

int ai_factory_price_black_scholes_american_put_cpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::black_scholes::price_american_put_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_black_scholes_american_put_gpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_black_scholes_american_put_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_black_scholes_asian_arithmetic_cpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::black_scholes::price_asian_arithmetic_call_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_black_scholes_asian_arithmetic_delta_crn_cpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::black_scholes::price_asian_arithmetic_call_delta_crn_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs
        );
    });
}

int ai_factory_price_black_scholes_asian_arithmetic_gpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_black_scholes_asian_arithmetic_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_black_scholes_asian_arithmetic_delta_crn_gpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_black_scholes_asian_arithmetic_delta_crn_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_black_scholes_volatility_swap_cpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::black_scholes::price_volatility_swap_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_black_scholes_volatility_swap_gpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_black_scholes_volatility_swap_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_black_scholes_lookback_fixed_cpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::black_scholes::price_lookback_fixed_call_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_black_scholes_lookback_fixed_delta_crn_cpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::black_scholes::price_lookback_fixed_call_delta_crn_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs
        );
    });
}

int ai_factory_price_black_scholes_lookback_fixed_gpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_black_scholes_lookback_fixed_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_black_scholes_lookback_fixed_delta_crn_gpu_batch(
    const ai_factory::cuda::BlackScholesRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_black_scholes_lookback_fixed_delta_crn_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_cpu_generate_black_scholes_spot_paths(
    const ai_factory::cuda::BlackScholesRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths
) {
    return call_with_error_boundary([&] {
        const ai_factory::simulation::BlackScholesModel model{
            row->spot,
            row->risk_free_rate,
            row->dividend_yield,
            row->volatility,
        };
        const ai_factory::simulation::TimeGrid time_grid{
            row->maturity,
            num_steps,
        };
        const ai_factory::simulation::SimulationConfig simulation{
            row->seed,
            num_paths,
            ai_factory::simulation::kPhilox4x32_10BoxMuller,
        };
        const auto paths = ai_factory::simulation::generate_black_scholes_spot_paths(
            model,
            time_grid,
            simulation
        );
        std::copy(paths.begin(), paths.end(), spot_paths);
    });
}

int ai_factory_generate_black_scholes_spot_paths(
    const ai_factory::cuda::BlackScholesRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::generate_black_scholes_spot_paths_cuda(
            row,
            num_paths,
            num_steps,
            spot_paths,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_american_put_cpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_american_put_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_heston_american_put_gpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_american_put_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_asian_arithmetic_delta_crn_cpu(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_asian_arithmetic_call_delta_crn(
            *row,
            num_paths,
            num_steps,
            relative_bump,
            *output
        );
    });
}

int ai_factory_price_heston_asian_arithmetic_cpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_asian_arithmetic_call_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_heston_asian_arithmetic_delta_crn_cpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_asian_arithmetic_call_delta_crn_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs
        );
    });
}

int ai_factory_price_heston_asian_arithmetic_gpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_asian_arithmetic_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_asian_arithmetic_delta_crn_gpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_asian_arithmetic_delta_crn_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_volatility_swap_cpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_volatility_swap_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_heston_volatility_swap_gpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_volatility_swap_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_lookback_fixed_cpu(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_lookback_fixed_call(
            *row,
            num_paths,
            num_steps,
            *output
        );
    });
}

int ai_factory_price_heston_lookback_fixed_delta_crn_cpu(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* output
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_lookback_fixed_call_delta_crn(
            *row,
            num_paths,
            num_steps,
            relative_bump,
            *output
        );
    });
}

int ai_factory_price_heston_lookback_fixed_cpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_lookback_fixed_call_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs
        );
    });
}

int ai_factory_price_heston_lookback_fixed_delta_crn_cpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs
) {
    return call_with_error_boundary([&] {
        ai_factory::cpu::heston::price_lookback_fixed_call_delta_crn_batch(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs
        );
    });
}

int ai_factory_price_heston_lookback_fixed_gpu(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* output,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_lookback_fixed_cuda(
            row,
            1U,
            num_paths,
            num_steps,
            output,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_lookback_fixed_delta_crn_gpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    double relative_bump,
    ai_factory::cuda::PriceDeltaOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_lookback_fixed_delta_crn_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            relative_bump,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

int ai_factory_price_heston_lookback_fixed_gpu_batch(
    const ai_factory::cuda::HestonRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs,
    double* kernel_seconds
) {
    return call_with_error_boundary([&] {
        ai_factory::cuda::CudaTiming timing{};
        ai_factory::cuda::price_heston_lookback_fixed_cuda(
            rows,
            row_count,
            num_paths,
            num_steps,
            outputs,
            &timing
        );
        if (kernel_seconds != nullptr) {
            *kernel_seconds = static_cast<double>(timing.total_ms) / 1000.0;
        }
    });
}

#define AI_FACTORY_TERMINAL_C_API(model, row_type, product, cpu_function, gpu_function) \
int ai_factory_price_##model##_##product##_cpu_batch( \
    const ai_factory::cuda::row_type* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, \
    ai_factory::cuda::MonteCarloOutput* outputs \
) { \
    return call_with_error_boundary([&] { \
        cpu_function(rows, row_count, num_paths, num_steps, outputs); \
    }); \
} \
int ai_factory_price_##model##_##product##_gpu_batch( \
    const ai_factory::cuda::row_type* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, \
    ai_factory::cuda::MonteCarloOutput* outputs, double* kernel_seconds \
) { \
    return call_with_error_boundary([&] { \
        ai_factory::cuda::CudaTiming timing{}; \
        gpu_function(rows, row_count, num_paths, num_steps, outputs, &timing); \
        if (kernel_seconds != nullptr) *kernel_seconds = timing.total_ms / 1000.0; \
    }); \
}

AI_FACTORY_TERMINAL_C_API(
    black_scholes, BlackScholesRow, european_call,
    ai_factory::cpu::black_scholes::price_european_call_batch,
    ai_factory::cuda::price_black_scholes_european_call_cuda
)
AI_FACTORY_TERMINAL_C_API(
    black_scholes, BlackScholesRow, digital_call,
    ai_factory::cpu::black_scholes::price_digital_call_batch,
    ai_factory::cuda::price_black_scholes_digital_call_cuda
)
AI_FACTORY_TERMINAL_C_API(
    heston, HestonRow, european_call,
    ai_factory::cpu::heston::price_european_call_batch,
    ai_factory::cuda::price_heston_european_call_cuda
)
AI_FACTORY_TERMINAL_C_API(
    heston, HestonRow, digital_call,
    ai_factory::cpu::heston::price_digital_call_batch,
    ai_factory::cuda::price_heston_digital_call_cuda
)
AI_FACTORY_TERMINAL_C_API(
    rough_bergomi, RoughBergomiRow, european_call,
    ai_factory::cpu::rough_bergomi::price_european_call_batch,
    ai_factory::cuda::price_rough_bergomi_european_call_cuda
)
AI_FACTORY_TERMINAL_C_API(
    rough_bergomi, RoughBergomiRow, digital_call,
    ai_factory::cpu::rough_bergomi::price_digital_call_batch,
    ai_factory::cuda::price_rough_bergomi_digital_call_cuda
)
AI_FACTORY_TERMINAL_C_API(
    rough_heston, RoughHestonRow, european_call,
    ai_factory::cpu::rough_heston::price_european_call_batch,
    ai_factory::cuda::price_rough_heston_european_call_cuda
)
AI_FACTORY_TERMINAL_C_API(
    rough_heston, RoughHestonRow, digital_call,
    ai_factory::cpu::rough_heston::price_digital_call_batch,
    ai_factory::cuda::price_rough_heston_digital_call_cuda
)

#undef AI_FACTORY_TERMINAL_C_API

int ai_factory_cpu_generate_heston_terminal_spots(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* terminal_spots
) {
    return call_with_error_boundary([&] {
        const ai_factory::simulation::HestonModel model{
            row->spot,
            row->risk_free_rate,
            row->dividend_yield,
            row->initial_variance,
            row->kappa,
            row->theta,
            row->volatility_of_variance,
            row->rho,
        };
        const ai_factory::simulation::TimeGrid time_grid{
            row->maturity,
            num_steps,
        };
        const ai_factory::simulation::SimulationConfig simulation{
            row->seed,
            num_paths,
            ai_factory::simulation::kPhilox4x32_10BoxMuller,
        };
        const auto spots = ai_factory::simulation::generate_heston_terminal_spots(
            model,
            time_grid,
            simulation,
            static_cast<ai_factory::simulation::HestonSimulationScheme>(
                row->scheme
            )
        );
        for (std::size_t index = 0; index < num_paths; ++index) {
            terminal_spots[index] = spots[index];
        }
    });
}

int ai_factory_cpu_generate_heston_max_spots(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* max_spots
) {
    return call_with_error_boundary([&] {
        const ai_factory::simulation::HestonModel model{
            row->spot,
            row->risk_free_rate,
            row->dividend_yield,
            row->initial_variance,
            row->kappa,
            row->theta,
            row->volatility_of_variance,
            row->rho,
        };
        const ai_factory::simulation::TimeGrid time_grid{
            row->maturity,
            num_steps,
        };
        const ai_factory::simulation::SimulationConfig simulation{
            row->seed,
            num_paths,
            ai_factory::simulation::kPhilox4x32_10BoxMuller,
        };
        const auto spots = ai_factory::simulation::generate_heston_max_spots(
            model,
            time_grid,
            simulation,
            static_cast<ai_factory::simulation::HestonSimulationScheme>(
                row->scheme
            )
        );
        for (std::size_t index = 0; index < num_paths; ++index) {
            max_spots[index] = spots[index];
        }
    });
}

int ai_factory_cpu_generate_heston_spot_paths(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths
) {
    return call_with_error_boundary([&] {
        const ai_factory::simulation::HestonModel model{
            row->spot,
            row->risk_free_rate,
            row->dividend_yield,
            row->initial_variance,
            row->kappa,
            row->theta,
            row->volatility_of_variance,
            row->rho,
        };
        const ai_factory::simulation::TimeGrid time_grid{
            row->maturity,
            num_steps,
        };
        const ai_factory::simulation::SimulationConfig simulation{
            row->seed,
            num_paths,
            ai_factory::simulation::kPhilox4x32_10BoxMuller,
        };
        const auto paths = ai_factory::simulation::generate_heston_spot_paths(
            model,
            time_grid,
            simulation,
            static_cast<ai_factory::simulation::HestonSimulationScheme>(
                row->scheme
            )
        );
        const std::size_t output_count = num_paths * (num_steps + 1U);
        for (std::size_t index = 0; index < output_count; ++index) {
            spot_paths[index] = paths[index];
        }
    });
}

int ai_factory_cpu_generate_heston_state_paths(
    const ai_factory::cuda::HestonRow* row,
    std::size_t num_paths,
    std::size_t num_steps,
    double* spot_paths,
    double* variance_paths
) {
    return call_with_error_boundary([&] {
        const ai_factory::simulation::HestonModel model{
            row->spot, row->risk_free_rate, row->dividend_yield,
            row->initial_variance, row->kappa, row->theta,
            row->volatility_of_variance, row->rho,
        };
        const ai_factory::simulation::TimeGrid time_grid{row->maturity, num_steps};
        const ai_factory::simulation::SimulationConfig simulation{
            row->seed, num_paths,
            ai_factory::simulation::kPhilox4x32_10BoxMuller,
        };
        const auto paths = ai_factory::simulation::generate_heston_state_paths(
            model, time_grid, simulation,
            static_cast<ai_factory::simulation::HestonSimulationScheme>(row->scheme)
        );
        std::copy(paths.spots.begin(), paths.spots.end(), spot_paths);
        std::copy(paths.variances.begin(), paths.variances.end(), variance_paths);
    });
}

#define AI_FACTORY_ROUGH_HESTON_PRICE_API(name, cpu_function, gpu_function) \
int ai_factory_price_rough_heston_##name##_cpu_batch( \
    const ai_factory::cuda::RoughHestonRow* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, \
    ai_factory::cuda::MonteCarloOutput* outputs \
) { return call_with_error_boundary([&]{ cpu_function(rows,row_count,num_paths,num_steps,outputs); }); } \
int ai_factory_price_rough_heston_##name##_gpu_batch( \
    const ai_factory::cuda::RoughHestonRow* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, \
    ai_factory::cuda::MonteCarloOutput* outputs, double* seconds \
) { return call_with_error_boundary([&]{ ai_factory::cuda::CudaTiming timing{}; \
    gpu_function(rows,row_count,num_paths,num_steps,outputs,&timing); \
    if(seconds)*seconds=timing.total_ms/1000.0; }); }

AI_FACTORY_ROUGH_HESTON_PRICE_API(
    asian_arithmetic,
    ai_factory::cpu::rough_heston::price_asian_arithmetic_call_batch,
    ai_factory::cuda::price_rough_heston_asian_arithmetic_cuda
)
AI_FACTORY_ROUGH_HESTON_PRICE_API(
    lookback_fixed,
    ai_factory::cpu::rough_heston::price_lookback_fixed_call_batch,
    ai_factory::cuda::price_rough_heston_lookback_fixed_cuda
)
AI_FACTORY_ROUGH_HESTON_PRICE_API(
    volatility_swap,
    ai_factory::cpu::rough_heston::price_volatility_swap_batch,
    ai_factory::cuda::price_rough_heston_volatility_swap_cuda
)
#undef AI_FACTORY_ROUGH_HESTON_PRICE_API

#define AI_FACTORY_ROUGH_HESTON_DELTA_API(name, cpu_function, gpu_function) \
int ai_factory_price_rough_heston_##name##_delta_crn_cpu_batch( \
    const ai_factory::cuda::RoughHestonRow* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, double bump, \
    ai_factory::cuda::PriceDeltaOutput* outputs \
) { return call_with_error_boundary([&]{ cpu_function(rows,row_count,num_paths,num_steps,bump,outputs); }); } \
int ai_factory_price_rough_heston_##name##_delta_crn_gpu_batch( \
    const ai_factory::cuda::RoughHestonRow* rows, std::size_t row_count, \
    std::size_t num_paths, std::size_t num_steps, double bump, \
    ai_factory::cuda::PriceDeltaOutput* outputs, double* seconds \
) { return call_with_error_boundary([&]{ ai_factory::cuda::CudaTiming timing{}; \
    gpu_function(rows,row_count,num_paths,num_steps,bump,outputs,&timing); \
    if(seconds)*seconds=timing.total_ms/1000.0; }); }

AI_FACTORY_ROUGH_HESTON_DELTA_API(
    asian_arithmetic,
    ai_factory::cpu::rough_heston::price_asian_arithmetic_call_delta_crn_batch,
    ai_factory::cuda::price_rough_heston_asian_arithmetic_delta_crn_cuda
)
AI_FACTORY_ROUGH_HESTON_DELTA_API(
    lookback_fixed,
    ai_factory::cpu::rough_heston::price_lookback_fixed_call_delta_crn_batch,
    ai_factory::cuda::price_rough_heston_lookback_fixed_delta_crn_cuda
)
#undef AI_FACTORY_ROUGH_HESTON_DELTA_API

int ai_factory_price_rough_heston_autocall_cpu_batch(
    const ai_factory::cuda::RoughHestonAutocallRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs
) { return call_with_error_boundary([&]{ai_factory::cpu::rough_heston::price_autocall_batch(rows,row_count,num_paths,num_steps,outputs);}); }
int ai_factory_price_rough_heston_autocall_gpu_batch(
    const ai_factory::cuda::RoughHestonAutocallRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    ai_factory::cuda::AutocallOutput* outputs, double* seconds
) { return call_with_error_boundary([&]{ai_factory::cuda::CudaTiming timing{};ai_factory::cuda::price_rough_heston_autocall_cuda(rows,row_count,num_paths,num_steps,outputs,&timing);if(seconds)*seconds=timing.total_ms/1000.0;}); }

int ai_factory_price_rough_heston_american_put_cpu_batch(
    const ai_factory::cuda::RoughHestonRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs
) { return call_with_error_boundary([&]{
    ai_factory::cpu::rough_heston::price_american_put_batch(
        rows, row_count, num_paths, num_steps, outputs
    );
}); }

int ai_factory_price_rough_heston_american_put_gpu_batch(
    const ai_factory::cuda::RoughHestonRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    ai_factory::cuda::MonteCarloOutput* outputs, double* seconds
) { return call_with_error_boundary([&]{
    ai_factory::cuda::CudaTiming timing{};
    ai_factory::cuda::price_rough_heston_american_put_cuda(
        rows, row_count, num_paths, num_steps, outputs, &timing
    );
    if (seconds) *seconds = timing.total_ms / 1000.0;
}); }

int ai_factory_cpu_generate_rough_heston_spot_paths(
    const ai_factory::cuda::RoughHestonRow* row,
    std::size_t num_paths, std::size_t num_steps, double* paths
) { return call_with_error_boundary([&]{
    const ai_factory::simulation::RoughHestonModel model{
        row->spot, row->risk_free_rate, row->dividend_yield,
        row->initial_variance, row->kappa, row->theta,
        row->volatility_of_variance, row->hurst, row->rho
    };
    const ai_factory::simulation::TimeGrid grid{row->maturity, num_steps};
    const ai_factory::simulation::SimulationConfig simulation{
        row->seed, num_paths, ai_factory::simulation::kPhilox4x32_10BoxMuller
    };
    const auto values = ai_factory::simulation::generate_rough_heston_spot_paths(
        model, grid, simulation
    );
    std::copy(values.begin(), values.end(), paths);
}); }

int ai_factory_generate_rough_heston_spot_paths(
    const ai_factory::cuda::RoughHestonRow* row,
    std::size_t num_paths, std::size_t num_steps, double* paths,
    double* seconds
) { return call_with_error_boundary([&]{
    ai_factory::cuda::CudaTiming timing{};
    ai_factory::cuda::generate_rough_heston_spot_paths_cuda(
        row, num_paths, num_steps, paths, &timing
    );
    if (seconds) *seconds = timing.total_ms / 1000.0;
}); }

int ai_factory_cpu_generate_rough_heston_state_paths(
    const ai_factory::cuda::RoughHestonRow* row,
    std::size_t num_paths, std::size_t num_steps,
    double* paths, double* factors
) { return call_with_error_boundary([&]{
    const ai_factory::simulation::RoughHestonModel model{
        row->spot, row->risk_free_rate, row->dividend_yield,
        row->initial_variance, row->kappa, row->theta,
        row->volatility_of_variance, row->hurst, row->rho
    };
    const ai_factory::simulation::TimeGrid grid{row->maturity, num_steps};
    const ai_factory::simulation::SimulationConfig simulation{
        row->seed, num_paths, ai_factory::simulation::kPhilox4x32_10BoxMuller
    };
    const auto values = ai_factory::simulation::generate_rough_heston_state_paths(
        model, grid, simulation
    );
    std::copy(values.spots.begin(), values.spots.end(), paths);
    std::copy(values.factors.begin(), values.factors.end(), factors);
}); }

int ai_factory_generate_rough_heston_state_paths(
    const ai_factory::cuda::RoughHestonRow* row,
    std::size_t num_paths, std::size_t num_steps,
    double* paths, double* factors, double* seconds
) { return call_with_error_boundary([&]{
    ai_factory::cuda::CudaTiming timing{};
    ai_factory::cuda::generate_rough_heston_state_paths_cuda(
        row, num_paths, num_steps, paths, factors, &timing
    );
    if (seconds) *seconds = timing.total_ms / 1000.0;
}); }

}  // extern "C"
