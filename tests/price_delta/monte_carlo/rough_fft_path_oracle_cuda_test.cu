// Compare actual paired FFT moments with independently prepared scalar paths.
#define AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT 8
#include "common/volterra/hybrid_fft_price_delta.cuh"
#include "common/volterra/fractional_hybrid_kernel.cuh"
#include "common/volterra/log_modulated_hybrid_kernel.cuh"
#include "common/volterra/fractional_resolvent_hybrid_kernel.cuh"
#include "model/equity/rough/rough_bergomi/dynamics_impl.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/dynamics_impl.cuh"
#include "model/equity/rough/rough_stein_stein/dynamics_impl.cuh"
#include "model/equity/rough/rough_sabr/dynamics_impl.cuh"
#include "product/european_option/pricing_policy.cuh"
#include "product/up_and_out_option/pricing_policy.cuh"
#include "product/cliquet/pricing_policy.cuh"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"
#include "tests/price_delta/cuda_test_support.cuh"
#include <bit>
#include <iostream>
#include <vector>

namespace {
using namespace ai_factory::workbench;
using price_delta_test::require;
using price_delta_test::DeviceArray;
namespace fft = volterra::hybrid_fft;
namespace delta = equity::price_delta;

template<typename Row>
__global__ void independent_payoffs(const Row* prepared, typename Row::Path::Parameters model,
    typename Row::Product::ProductParameters product, const float* variances,
    delta::SpotBumpConfiguration configuration, float* outputs) {
    using Path = typename Row::Path;
    using Product = typename Row::Product;
    const auto row = *prepared;
    const auto bump = delta::prepare_spot_bump(model.spot, configuration);
    float payoffs[3];
    for (unsigned scenario=0;scenario<3;++scenario) {
        auto parameters=model;
        parameters.spot=scenario==0?bump.central:scenario==1?bump.lower:bump.upper;
        const auto dynamics=Path::prepare_model(parameters,row.schedule.time_step);
        const auto payoff=Product::prepare_product(parameters,product,
            equity::ProductPreparationContext{1.f/252.f,row.schedule.maturity_years});
        auto handler=Product::make_handler(payoff);
        equity::PathProductObservationAdapter<Path, decltype(handler), Product::kObservationCoordinate> observer{handler};
        const auto state=fft::simulate_observed_path<Path>(row,dynamics,threadIdx.x,
            variances,fft::DirectPathConvolution{},observer);
        payoffs[scenario]=Product::template finalize<Path>(payoff,state,handler);
    }
    outputs[2*threadIdx.x]=payoffs[0];
    outputs[2*threadIdx.x+1]=(payoffs[2]-payoffs[1])/bump.width;
}

template<typename Row>
__global__ void direct_convolutions(const Row* prepared,float2* convolutions) {
    const auto row=*prepared;
    const auto pair=threadIdx.x;
    for(unsigned step=0;step<row.schedule.step_count;++step)
        convolutions[pair*row.schedule.step_count+step]={
            fft::DirectPathConvolution{}.value(row,2*pair,step+1),
            fft::DirectPathConvolution{}.value(row,2*pair+1,step+1)};
}

template<typename Kernel, typename Path, typename Paths, typename Product, typename Schedule>
void check(const char* name, typename Path::Parameters model,
           typename Product::ProductParameters product) {
    using Row=fft::PreparedRow<Kernel,Path,Product,Schedule>;
    DeviceArray<typename Path::Parameters> dm(1);
    DeviceArray<typename Product::ProductParameters> dp(1);
    DeviceArray<Row> row(1);
    DeviceArray<float> variances(8), oracle(512), result(4);
    DeviceArray<float2> convolutions(128*8);
    DeviceArray<fft::PartialMoments> moments(2);
    check_cuda(cudaMemcpy(dm.data,&model,sizeof(model),cudaMemcpyHostToDevice),"model");
    check_cuda(cudaMemcpy(dp.data,&product,sizeof(product),cudaMemcpyHostToDevice),"product");
    fft::prepare_direct_row_kernel<Kernel,Path,Product,Schedule><<<1,256>>>(
        dm.data,dp.data,1,PriceConstruction::Aligned,0,8,{1.f/252.f,1.f/504.f},731,
        variances.data,row.data);
    direct_convolutions<Row><<<1,128>>>(row.data,convolutions.data);
    const dim3 grid(1), block(volterra::tuning::kPricingPathThreads);
    constexpr std::size_t shared=2U*(volterra::tuning::kPricingPathThreads/32U)*sizeof(double);
    report_cuda_kernel_phase_launch_if_enabled(name,"paired","path_evaluation",
        fft::evaluate_price_delta_paths_kernel<Row,Paths,fft::FftPathConvolution>,grid,block,shared);
    for(float width:{.01f,.005f}) {
        const delta::SpotBumpConfiguration bump{width};
        independent_payoffs<Row><<<1,256>>>(row.data,model,product,variances.data,bump,oracle.data);
        fft::evaluate_price_delta_paths_kernel<Row,Paths,fft::FftPathConvolution><<<grid,block,shared>>>(
            0,256,row.data,dm.data,dp.data,bump,1.f/252.f,variances.data,convolutions.data,
            moments.data,moments.data+1);
        fft::PricePathConsumer{}.finish(moments.data,256,0,result.data,result.data+1,name,"price");
        fft::PricePathConsumer{}.finish(moments.data+1,256,0,result.data+2,result.data+3,name,"delta");
        float paths[512], actual[4];
        check_cuda(cudaMemcpy(paths,oracle.data,sizeof(paths),cudaMemcpyDeviceToHost),"oracle");
        check_cuda(cudaMemcpy(actual,result.data,sizeof(actual),cudaMemcpyDeviceToHost),"moments");
        fft::evaluate_price_delta_paths_kernel<Row,Paths,fft::DirectPathConvolution><<<grid,block,shared>>>(
            0,256,row.data,dm.data,dp.data,bump,1.f/252.f,variances.data,nullptr,
            moments.data,moments.data+1);
        fft::PricePathConsumer{}.finish(moments.data,256,0,result.data,result.data+1,name,"direct price");
        fft::PricePathConsumer{}.finish(moments.data+1,256,0,result.data+2,result.data+3,name,"direct delta");
        float direct[4];
        check_cuda(cudaMemcpy(direct,result.data,sizeof(direct),cudaMemcpyDeviceToHost),"direct");
        for(unsigned i=0;i<4;++i)
            require(std::bit_cast<unsigned>(direct[i])==std::bit_cast<unsigned>(actual[i]),
                    "direct/convolution paired bits differ");
        for(unsigned component=0;component<2;++component) {
            double sum=0,squares=0;
            for(unsigned i=0;i<256;++i) {double x=paths[2*i+component];sum+=x;squares+=x*x;}
            const double mean=sum/256.;
            const double error=std::sqrt(std::max(0.,(squares-sum*sum/256.)/(255.*256.)));
            const double tolerance=component==0?1e-6:3e-4;
            require(std::abs(actual[2*component]-mean)<tolerance,"independent paired mean mismatch");
            require(std::abs(actual[2*component+1]-error)<tolerance,"independent paired SE mismatch");
        }
    }
    std::cout<<name<<": independent scalar paths, paired mean and standard error\n";
}

template<typename Kernel, typename Path, typename Paths>
void model_cases(const char* name, typename Path::Parameters model) {
    check<Kernel,Path,Paths,product::EuropeanOptionPathPolicy<OptionSide::call>,volterra::TerminalHybridSchedule>(
        (std::string(name)+" terminal").c_str(),model,{1.2f,4U});
    check<Kernel,Path,Paths,product::UpAndOutOptionPathPolicy<OptionSide::call>,volterra::DenseHybridSchedule>(
        (std::string(name)+" barrier").c_str(),model,{1.1f,1.202f,4U});
    check<Kernel,Path,Paths,product::CliquetPathPolicy,volterra::RegularHybridSchedule>(
        (std::string(name)+" cliquet").c_str(),model,{4U,1U,1.f,-.1f,.1f,-.3f,.3f});
    check<Kernel,Path,Paths,product::PhoenixMemoryAutocallPathPolicy,volterra::RegularHybridSchedule>(
        (std::string(name)+" phoenix memory").c_str(),model,{4U,1U,1.002f,.98f,.8f,.1f});
}
}
int main() {
    using namespace ai_factory::workbench;
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
    try {
        using B=model::equity::rough_bergomi::PathPolicy;
        using L=model::equity::log_modulated_rough_bergomi::PathPolicy;
        using S=model::equity::rough_stein_stein::PathPolicy;
        using R=model::equity::rough_sabr::PathPolicy;
        model_cases<volterra::FractionalHybridKernelPolicy,B,delta::MultiplicativeVolterraSpotPath<B>>(
            "Bergomi",{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f});
        model_cases<volterra::LogModulatedHybridKernelPolicy,L,delta::MultiplicativeVolterraSpotPath<L>>(
            "log Bergomi",{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f,1.f,1.5f});
        model_cases<volterra::FractionalResolventHybridKernelPolicy,S,delta::MultiplicativeVolterraSpotPath<S>>(
            "Stein Stein",{1.2f,.03f,.01f,.2f,1.5f,.3f,.1f,-.7f});
        model_cases<volterra::FractionalHybridKernelPolicy,R,delta::CoupledVolterraSpotPaths<R>>(
            "SABR",{1.2f,.03f,.01f,.04f,1.5f,.1f,-.7f,.6f});
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
