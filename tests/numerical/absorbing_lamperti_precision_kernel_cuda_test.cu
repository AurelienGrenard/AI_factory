// Resolve near-beta-one Lamperti increments without FP32 cancellation.
#include "common/check_cuda.cuh"
#include "common/equity/absorbing_lamperti.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace {
constexpr float beta=0.9993257522583008f;
constexpr float spot=1.8213049173355103f;
constexpr float dt=1.0f/504.0f;
constexpr float drift=(-0.021024275571107865f)*dt;
constexpr float normals[5]{-3.0f,-1.0f,0.0f,1.0f,3.0f};
__global__ void evaluate(float* output) {
    if (threadIdx.x!=0U || blockIdx.x!=0U) return;
    const float q=1.0f-beta;
    const float volatility=sqrtf(0.03999999910593033f)*powf(spot,q);
    const float device_normals[5]{-3.0f,-1.0f,0.0f,1.0f,3.0f};
    for (int i=0;i<5;++i) {
        float log_spot=logf(spot);
        ai_factory::workbench::equity::advance_absorbing_lamperti_log_spot(
            beta,q,drift,dt,sqrtf(dt),volatility,device_normals[i],log_spot);
        output[i]=log_spot;
    }
    float absorbed=logf(1.0e-4f);
    ai_factory::workbench::equity::advance_absorbing_lamperti_log_spot(
        0.5f,0.5f,0.0f,1.0f,1.0f,1.0e6f,-1.0f,absorbed);
    output[5]=absorbed;
}
}
int main() {
    float* device=nullptr;
    ai_factory::workbench::check_cuda(cudaMalloc(&device,6*sizeof(float)),"Lamperti output allocation");
    evaluate<<<1,1>>>(device);
    ai_factory::workbench::check_cuda(cudaGetLastError(),"Lamperti launch");
    float actual[6]{};
    ai_factory::workbench::check_cuda(cudaMemcpy(actual,device,sizeof(actual),cudaMemcpyDeviceToHost),"Lamperti readback");
    ai_factory::workbench::check_cuda(cudaFree(device),"Lamperti free");
    const double q=static_cast<double>(1.0f-beta);
    const double x=std::log(static_cast<double>(spot));
    const double volatility=std::sqrt(static_cast<double>(0.03999999910593033f))
        *std::pow(static_cast<double>(spot),q);
    for (int i=0;i<5;++i) {
        const double power=std::exp(q*x);
        const double relative=q*(static_cast<double>(drift)
            +volatility*std::sqrt(static_cast<double>(dt))*normals[i]/power
            -0.5*static_cast<double>(beta)*volatility*volatility*dt/(power*power));
        const double reference=x+std::log1p(relative)/q;
        if (std::fabs(actual[i]-reference)>2.0e-6)
            throw std::runtime_error("Near-beta-one Lamperti step lost FP32 precision.");
    }
    if (actual[5]!=-INFINITY)
        throw std::runtime_error("Nonpositive Lamperti proposal did not absorb.");
}
