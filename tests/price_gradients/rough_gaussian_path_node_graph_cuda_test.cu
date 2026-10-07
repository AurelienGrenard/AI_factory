// Gaussian rough Asian graph: cached FFT, path payoff, CRN and price parity.
#include "common/check_cuda.cuh"
#include "common/volterra/hybrid_fft_workspace.cuh"
#ifndef ROUGH_GAUSSIAN_PATH_TEST_MODEL
#define ROUGH_GAUSSIAN_PATH_TEST_MODEL 0
#endif
#if ROUGH_GAUSSIAN_PATH_TEST_MODEL == 0
#include "model/equity/rough/rough_bergomi/product/asian_option.cuh"
#include "model/equity/rough/rough_bergomi/product/asian_option_price_gradients.cuh"
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 1
#include "model/equity/rough/rough_sabr/product/asian_option.cuh"
#include "model/equity/rough/rough_sabr/product/asian_option_price_gradients.cuh"
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 2
#include "model/equity/rough/rough_stein_stein/product/asian_option.cuh"
#include "model/equity/rough/rough_stein_stein/product/asian_option_price_gradients.cuh"
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 3
#include "model/equity/rough/log_modulated_rough_bergomi/product/asian_option.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/asian_option_price_gradients.cuh"
#else
#error "Unknown Gaussian rough path model"
#endif

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace mcpg = monte_carlo::price_gradients;
#if ROUGH_GAUSSIAN_PATH_TEST_MODEL == 0
namespace rb = model::equity::rough_bergomi;
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 1
namespace rb = model::equity::rough_sabr;
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 2
namespace rb = model::equity::rough_stein_stein;
#else
namespace rb = model::equity::log_modulated_rough_bergomi;
#endif
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<typename T>
struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(std::size_t n) { check_cuda(cudaMalloc(&ptr, n*sizeof(T)), "FFT graph alloc"); }
    ~Buffer() { cudaFree(ptr); }
    void upload(const T* x, std::size_t n) {
        check_cuda(cudaMemcpy(ptr,x,n*sizeof(T),cudaMemcpyHostToDevice),"FFT graph upload");
    }
    void download(T* x, std::size_t n) {
        check_cuda(cudaMemcpy(x,ptr,n*sizeof(T),cudaMemcpyDeviceToHost),"FFT graph download");
    }
};
}
int main() {
    int devices=0;
    const auto availability=cudaGetDeviceCount(&devices);
    if (availability==cudaErrorNoDevice
        || availability==cudaErrorInsufficientDriver || devices==0) return 77;
    check_cuda(availability,"FFT graph cudaGetDeviceCount");
    constexpr std::size_t paths=513U;
    constexpr std::uint64_t seed=6734900U;
#if ROUGH_GAUSSIAN_PATH_TEST_MODEL == 0
    const std::vector<rb::ModelParameters> models{{
        1.0f,.02f,.01f,.04f,.8f,.10f,-.7f
    }};
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 1
    const std::vector<rb::ModelParameters> models{{
        1.0f,.02f,.01f,.04f,.8f,.10f,-.7f,.85f
    }};
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 2
    const std::vector<rb::ModelParameters> models{{
        1.0f,.02f,.01f,.20f,.30f,.40f,.10f,-.7f
    }};
#else
    const std::vector<rb::ModelParameters> models{{
        1.0f,.02f,.01f,.04f,.8f,.10f,-.7f,1.0f,2.0f
    }};
#endif
    const std::vector<product::AsianOptionParameters> products{{1.0f,63U}};
    const pg::PriceGradientConfiguration configuration{{
        {"model.spot",{.01f,pg::BumpScale::relative}},
        {"product.strike",{.01f,pg::BumpScale::relative}}
    }};
    const auto host=
#if ROUGH_GAUSSIAN_PATH_TEST_MODEL == 0
        rb::prepare_rough_bergomi_asian_option_sensitivities(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 1
        rb::prepare_rough_sabr_asian_option_sensitivities(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 2
        rb::prepare_rough_stein_stein_asian_option_sensitivities(
#else
        rb::prepare_log_modulated_rough_bergomi_asian_option_sensitivities(
#endif
        models,products,PriceConstruction::CartesianProduct,
        {1.0f/504.0f,2U},configuration,
        {pg::SensitivityOrders::first_and_second});
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,0U,1U,paths,256U,1U,seed
    };
    using Graph=rb::AsianOptionNodeGraph<
        OptionSide::call,pg::SensitivityOrders::first_and_second,2U>;
    Graph::Cache cache;
    const auto plan=Graph::plan(host,cache,launch);
    require(cache.entries().size()==1U,
        "Spot and strike nodes duplicated the FFT kernel.");
    Buffer<unsigned char> workspace(plan.bytes);
    Buffer<float> prices(1),errors(1),gradients(2),gradient_errors(2),
        hessians(2),hessian_errors(2);
    Buffer<pg::SensitivityStencil<4U>> stencils(2);
    pg::SensitivityOutputs outputs{
        prices.ptr,errors.ptr,gradients.ptr,gradient_errors.ptr,
        hessians.ptr,hessian_errors.ptr,1U,2U
    };
    Graph::launch(host,cache,launch,outputs,
        mcpg::DevicePreparedStencilOutputs<4U>{stencils.ptr,2U,nullptr},
        plan,workspace.ptr,plan.bytes);
    check_cuda(cudaDeviceSynchronize(),"FFT path graph sync");
    float graph_price=0.0f, graph_gradients[2]{}, graph_hessians[2]{};
    prices.download(&graph_price,1);
    gradients.download(graph_gradients,2);
    hessians.download(graph_hessians,2);
    require(std::isfinite(graph_price),"Invalid FFT Asian graph price.");
    for (int i=0;i<2;++i)
        require(std::isfinite(graph_gradients[i])
            && std::isfinite(graph_hessians[i]),
            "Invalid FFT Asian graph derivative.");
    const auto first_host =
#if ROUGH_GAUSSIAN_PATH_TEST_MODEL == 0
        rb::prepare_rough_bergomi_asian_option_sensitivities(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 1
        rb::prepare_rough_sabr_asian_option_sensitivities(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 2
        rb::prepare_rough_stein_stein_asian_option_sensitivities(
#else
        rb::prepare_log_modulated_rough_bergomi_asian_option_sensitivities(
#endif
        models, products, PriceConstruction::CartesianProduct,
        {1.0f/504.0f, 2U}, configuration,
        {pg::SensitivityOrders::first});
    using FirstGraph = rb::AsianOptionNodeGraph<
        OptionSide::call, pg::SensitivityOrders::first, 2U>;
    FirstGraph::Cache first_cache;
    const auto first_plan = FirstGraph::plan(first_host, first_cache, launch);
    Buffer<unsigned char> first_workspace(first_plan.bytes);
    FirstGraph::launch(first_host, first_cache, launch,
        {prices.ptr, errors.ptr, gradients.ptr, gradient_errors.ptr,
         nullptr, nullptr, 1U, 2U},
        mcpg::DevicePreparedStencilOutputs<4U>{stencils.ptr, 2U, nullptr},
        first_plan, first_workspace.ptr, first_plan.bytes);
    check_cuda(cudaDeviceSynchronize(), "first-only FFT path graph sync");
    float first_price = 0.0f, first_gradients[2]{};
    prices.download(&first_price, 1);
    gradients.download(first_gradients, 2);
    require(std::abs(first_price - graph_price) < 5.0e-5f,
            "First-only FFT graph changed the Asian price.");
    for (int i=0; i<2; ++i)
        require(std::abs(first_gradients[i] - graph_gradients[i]) < 1.0e-4f,
                "First-only FFT graph changed an Asian delta.");
    const auto bytes=volterra::required_hybrid_fft_workspace_bytes(
        126U,paths,512U);
    Buffer<unsigned char> reference_workspace(bytes);
    Buffer<rb::ModelParameters> dm(1);
    Buffer<product::AsianOptionParameters> dp(1);
    Buffer<float> reference_prices(1),reference_errors(1);
    dm.upload(models.data(),1);
    dp.upload(products.data(),1);
#if ROUGH_GAUSSIAN_PATH_TEST_MODEL == 0
    rb::launch_rough_bergomi_asian_option_cuda<OptionSide::call>(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 1
    rb::launch_rough_sabr_asian_option_cuda<OptionSide::call>(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 2
    rb::launch_rough_stein_stein_asian_option_cuda<OptionSide::call>(
#else
    rb::launch_log_modulated_rough_bergomi_asian_option_cuda<OptionSide::call>(
#endif
        dm.ptr,1U,dp.ptr,1U,PriceConstruction::CartesianProduct,
        1U,0U,paths,1.0f/252.0f,1.0f/504.0f,126U,512U,
        reference_workspace.ptr,bytes,seed,
        reference_prices.ptr,reference_errors.ptr);
    check_cuda(cudaDeviceSynchronize(),"FFT Asian reference sync");
    float expected=0.0f;
    reference_prices.download(&expected,1);
    require(std::abs(graph_price-expected)<5.0e-5f,
        "FFT Asian graph price differs from the existing pricer.");
    float endpoint_prices[2]{};
    for (std::size_t endpoint=0U; endpoint<2U; ++endpoint) {
        auto bumped=models[0U];
        bumped.spot *= endpoint==0U ? .99f : 1.01f;
        dm.upload(&bumped,1);
#if ROUGH_GAUSSIAN_PATH_TEST_MODEL == 0
        rb::launch_rough_bergomi_asian_option_cuda<OptionSide::call>(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 1
        rb::launch_rough_sabr_asian_option_cuda<OptionSide::call>(
#elif ROUGH_GAUSSIAN_PATH_TEST_MODEL == 2
        rb::launch_rough_stein_stein_asian_option_cuda<OptionSide::call>(
#else
        rb::launch_log_modulated_rough_bergomi_asian_option_cuda<OptionSide::call>(
#endif
            dm.ptr,1U,dp.ptr,1U,PriceConstruction::CartesianProduct,
            1U,0U,paths,1.0f/252.0f,1.0f/504.0f,126U,512U,
            reference_workspace.ptr,bytes,seed,
            reference_prices.ptr,reference_errors.ptr);
        check_cuda(cudaDeviceSynchronize(),"FFT Asian CRN reference sync");
        reference_prices.download(&endpoint_prices[endpoint],1);
    }
    const float oracle_delta=(endpoint_prices[1]-endpoint_prices[0])/.02f;
    require(std::abs(graph_gradients[0]-oracle_delta)<3.0e-3f,
            "FFT Asian graph spot delta differs from paired FFT prices.");
}
