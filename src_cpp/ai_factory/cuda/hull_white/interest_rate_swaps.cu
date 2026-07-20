#include "ai_factory/cuda/hull_white/interest_rate_swaps.cuh"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"
#include "ai_factory/cuda/common/runtime.cuh"
namespace ai_factory::cuda { namespace {
struct WorkspaceTag {};
__global__ void kernel(const HullWhiteSwapRow* rows,std::size_t count,MonteCarloOutput* outputs){
    const std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=count)return; const auto row=rows[i]; const auto& p=row.product;
    double annuity=0.0;
    for(int payment=1;payment<=p.payment_count;++payment){const double t=p.start_time+payment*p.accrual_period;annuity+=p.accrual_period*fixed_income::nelson_siegel_discount(t,row.beta0,row.beta1,row.beta2,row.tau);}
    const double end=p.start_time+p.payment_count*p.accrual_period;
    const double value=p.notional*static_cast<double>(p.direction)*(fixed_income::nelson_siegel_discount(p.start_time,row.beta0,row.beta1,row.beta2,row.tau)-fixed_income::nelson_siegel_discount(end,row.beta0,row.beta1,row.beta2,row.tau)-p.fixed_rate*annuity);
    outputs[i]={value,0.0};
}
}
void price_hull_white_interest_rate_swap_cuda(const HullWhiteSwapRow* host,std::size_t count,MonteCarloOutput* out,CudaTiming* timing){
    auto& w=detail::reusable_cuda_workspace<WorkspaceTag,2U>(); auto* rows=w.buffer<HullWhiteSwapRow>(0,count,"cudaMalloc rows");auto* outputs=w.buffer<MonteCarloOutput>(1,count,"cudaMalloc outputs");
    detail::check_cuda(cudaMemcpy(rows,host,count*sizeof(*host),cudaMemcpyHostToDevice),"cudaMemcpy rows");auto start=w.start_event(),stop=w.stop_event();detail::check_cuda(cudaEventRecord(start),"event start");kernel<<<static_cast<unsigned>((count+255U)/256U),256>>>(rows,count,outputs);detail::check_cuda(cudaGetLastError(),"Hull-White swap kernel");detail::check_cuda(cudaEventRecord(stop),"event stop");detail::check_cuda(cudaEventSynchronize(stop),"event sync");float ms=0;detail::check_cuda(cudaEventElapsedTime(&ms,start,stop),"event elapsed");detail::check_cuda(cudaMemcpy(out,outputs,count*sizeof(*out),cudaMemcpyDeviceToHost),"cudaMemcpy outputs");if(timing){timing->simulation_ms=ms;timing->total_ms=ms;}
}
}
