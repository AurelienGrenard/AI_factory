// Asian path graph must preserve the existing prepared-lift price and CRN moments.
#include "common/check_cuda.cuh"
#include "model/equity/rough/rough_heston/product/asian_option.cuh"
#include "model/equity/rough/rough_heston/product/asian_option_price_gradients.cuh"
#include "model/equity/rough/rough_heston/product/cliquet_price_gradients.cuh"
#include "model/equity/rough/rough_heston/product/forward_start_option_price_gradients.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/asian_option.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/asian_option_price_gradients.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace mcpg = monte_carlo::price_gradients;
namespace rh = model::equity::rough_heston;
namespace qrh = model::equity::quadratic_rough_heston;

void require(bool yes, const char* why) { if (!yes) throw std::runtime_error(why); }

template<typename T>
struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(std::size_t n) { check_cuda(cudaMalloc(&ptr, n*sizeof(T)), "path alloc"); }
    ~Buffer() { cudaFree(ptr); }
    void upload(const T* x, std::size_t n) {
        check_cuda(cudaMemcpy(ptr, x, n*sizeof(T), cudaMemcpyHostToDevice), "path upload");
    }
    void download(T* x, std::size_t n) {
        check_cuda(cudaMemcpy(x, ptr, n*sizeof(T), cudaMemcpyDeviceToHost), "path download");
    }
};

template<typename Model, typename Prepared, typename Cache, typename Graph,
         typename FirstGraph, typename Prepare, typename Reference>
void run(const Model& model, Prepare prepare, Reference reference) {
    constexpr float dt = 1.0f / 504.0f;
    constexpr std::size_t paths = 513U;
    constexpr std::uint64_t seed = 6734900U;
    const std::vector<Model> models{model};
    const std::vector<product::AsianOptionParameters> products{{1.0f, 63U}};
    const pg::PriceGradientConfiguration configuration{{
        {"model.spot", {.01f, pg::BumpScale::relative}},
        {"product.strike", {.01f, pg::BumpScale::relative}}
    }};
    const auto host = prepare(models, products, PriceConstruction::CartesianProduct,
        pg::TimeConfiguration{dt, 2U}, configuration,
        pg::SensitivityRequest{pg::SensitivityOrders::first_and_second});
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, 1U, paths, 256U, 1U, seed
    };
    Cache cache(1.0f, dt);
    const auto plan = Graph::plan(host, cache, launch);
    require(cache.values().size() == 1U,
            "Spot and strike nodes must share lift dynamics.");
    Buffer<unsigned char> workspace(plan.bytes);
    Buffer<float> prices(1), errors(1), gradients(2), gradient_errors(2),
        hessians(2), hessian_errors(2);
    Buffer<pg::SensitivityStencil<4U>> stencils(2);
    pg::SensitivityOutputs outputs{
        prices.ptr, errors.ptr, gradients.ptr, gradient_errors.ptr,
        hessians.ptr, hessian_errors.ptr, 1U, 2U
    };
    Graph::launch(host, cache, launch, outputs,
        mcpg::DevicePreparedStencilOutputs<4U>{stencils.ptr, 2U, nullptr},
        plan, workspace.ptr, plan.bytes);
    check_cuda(cudaDeviceSynchronize(), "path graph sync");
    float got = 0.0f, grads[2]{}, diagonals[2]{};
    prices.download(&got, 1);
    gradients.download(grads, 2);
    hessians.download(diagonals, 2);
    for (int i=0; i<2; ++i)
        require(std::isfinite(grads[i]) && std::isfinite(diagonals[i]),
                "Invalid Asian graph sensitivity.");
    const auto first_host = prepare(
        models, products, PriceConstruction::CartesianProduct,
        pg::TimeConfiguration{dt, 2U}, configuration,
        pg::SensitivityRequest{pg::SensitivityOrders::first});
    Cache first_cache(1.0f, dt);
    const auto first_plan = FirstGraph::plan(first_host, first_cache, launch);
    Buffer<unsigned char> first_workspace(first_plan.bytes);
    FirstGraph::launch(first_host, first_cache, launch,
        {prices.ptr, errors.ptr, gradients.ptr, gradient_errors.ptr,
         nullptr, nullptr, 1U, 2U},
        mcpg::DevicePreparedStencilOutputs<4U>{stencils.ptr, 2U, nullptr},
        first_plan, first_workspace.ptr, first_plan.bytes);
    check_cuda(cudaDeviceSynchronize(), "first-only path graph sync");
    float first_price = 0.0f, first_gradients[2]{};
    prices.download(&first_price, 1);
    gradients.download(first_gradients, 2);
    require(std::abs(first_price - got) < 1.0e-6f,
            "First-only lift graph changed the Asian price.");
    for (int i=0; i<2; ++i)
        require(std::abs(first_gradients[i] - grads[i]) < 1.0e-4f,
                "First-only lift graph changed an Asian delta.");
    Buffer<Model> dm(1);
    Buffer<Prepared> dd(1);
    Buffer<product::AsianOptionParameters> dp(1);
    Buffer<float> reference_price(1), reference_error(1);
    dm.upload(&model, 1);
    dd.upload(&cache.values()[0], 1);
    dp.upload(products.data(), 1);
    reference(dm.ptr, dd.ptr, products.data(), dp.ptr,
              paths, seed, reference_price.ptr, reference_error.ptr);
    check_cuda(cudaDeviceSynchronize(), "path reference sync");
    float expected = 0.0f;
    reference_price.download(&expected, 1);
    require(std::abs(got-expected) < 1.0e-6f,
            "Asian graph price differs from existing lift pricer.");
    float endpoint_prices[2]{};
    Cache oracle_cache(1.0f, dt);
    for (std::size_t endpoint=0U; endpoint<2U; ++endpoint) {
        auto bumped=model;
        bumped.spot *= endpoint==0U ? .99f : 1.01f;
        const auto index=oracle_cache.index(bumped);
        dm.upload(&bumped,1);
        dd.upload(&oracle_cache.values()[index],1);
        reference(dm.ptr, dd.ptr, products.data(), dp.ptr,
                  paths, seed, reference_price.ptr, reference_error.ptr);
        check_cuda(cudaDeviceSynchronize(),"Asian CRN reference sync");
        reference_price.download(&endpoint_prices[endpoint],1);
    }
    const float oracle_delta=(endpoint_prices[1]-endpoint_prices[0])
        / (.02f*model.spot);
    require(std::abs(grads[0]-oracle_delta)<3.0e-3f,
            "Asian graph spot delta differs from paired lift prices.");
}

template<typename Graph, typename Product, typename Prepare>
void run_maturity_graph(const Product& product, Prepare prepare) {
    constexpr float dt = 1.0f / 504.0f;
    const std::vector<rh::ModelParameters> models{{
        1.0f, .02f, .01f, .04f, .30f, .02f, .30f, .10f, -.70f
    }};
    const std::vector<Product> products{product};
    const pg::PriceGradientConfiguration bump{{
        {"product.maturity_years", {dt, pg::BumpScale::absolute}}
    }};
    const auto host = prepare(
        models, products, PriceConstruction::CartesianProduct,
        pg::TimeConfiguration{dt, 2U}, bump,
        pg::SensitivityRequest{pg::SensitivityOrders::first_and_second}
    );
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, 1U, 257U, 256U, 1U, 6734900U
    };
    rh::price_gradients::PreparedLiftNodeCache<7U> cache(1.0f, dt);
    const auto plan = Graph::plan(host, cache, launch);
    Buffer<unsigned char> workspace(plan.bytes);
    Buffer<float> prices(1U), errors(1U), gradients(1U),
        gradient_errors(1U), hessians(1U), hessian_errors(1U);
    Buffer<pg::SensitivityStencil<4U>> stencils(1U);
    Graph::launch(
        host, cache, launch,
        {prices.ptr, errors.ptr, gradients.ptr, gradient_errors.ptr,
         hessians.ptr, hessian_errors.ptr, 1U, 1U},
        mcpg::DevicePreparedStencilOutputs<4U>{stencils.ptr, 1U, nullptr},
        plan, workspace.ptr, plan.bytes
    );
    check_cuda(cudaDeviceSynchronize(), "lift maturity path graph sync");
    float price = 0.0f, gradient = 0.0f, hessian = 0.0f;
    prices.download(&price, 1U);
    gradients.download(&gradient, 1U);
    hessians.download(&hessian, 1U);
    require(std::isfinite(price) && std::isfinite(gradient)
        && std::isfinite(hessian),
        "Invalid lift path graph maturity node.");
}
}

int main() {
    int count=0;
    const auto availability = cudaGetDeviceCount(&count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver || count == 0) return 77;
    check_cuda(availability, "path cudaGetDeviceCount");
    run_maturity_graph<rh::ForwardStartOptionNodeGraph<
            OptionSide::call, 7U,
            pg::SensitivityOrders::first_and_second, 1U>>(
        product::ForwardStartOptionParameters{1.0f, 126U, 252U},
        rh::prepare_rough_heston_forward_start_option_sensitivities
    );
    run_maturity_graph<rh::CliquetNodeGraph<
            7U, pg::SensitivityOrders::first_and_second, 1U>>(
        product::CliquetParameters{252U, 21U, 1.0f, -.1f, .1f, -.2f, .2f},
        rh::prepare_rough_heston_cliquet_sensitivities
    );
    run<rh::ModelParameters, rh::PreparedDynamics<7U>,
        rh::price_gradients::PreparedLiftNodeCache<7U>,
        rh::AsianOptionNodeGraph<OptionSide::call, 7U,
            pg::SensitivityOrders::first_and_second, 2U>,
        rh::AsianOptionNodeGraph<OptionSide::call, 7U,
            pg::SensitivityOrders::first, 2U>>(
        rh::ModelParameters{1.0f,.02f,.01f,.04f,.30f,.02f,.30f,.10f,-.70f},
        rh::prepare_rough_heston_asian_option_sensitivities,
        [](auto dm, auto dd, auto hp, auto dp, auto paths, auto seed, auto p, auto e) {
            rh::launch_rough_heston_asian_option_cuda<OptionSide::call,7U>(
                dm,1U,dd,1U,hp,dp,1U,PriceConstruction::CartesianProduct,
                1U,0U,1U,paths,1.0f/504.0f,2U,256U,1U,seed,p,e);
        });
    run<qrh::ModelParameters, qrh::PreparedDynamics<7U>,
        qrh::price_gradients::PreparedLiftNodeCache<7U>,
        qrh::AsianOptionNodeGraph<OptionSide::call, 7U,
            pg::SensitivityOrders::first_and_second, 2U>,
        qrh::AsianOptionNodeGraph<OptionSide::call, 7U,
            pg::SensitivityOrders::first, 2U>>(
        qrh::ModelParameters{1.0f,.02f,.01f,.15f,.60f,.08f,.005f,1.50f,1.00f,.105371f},
        qrh::prepare_quadratic_rough_heston_asian_option_sensitivities,
        [](auto dm, auto dd, auto hp, auto dp, auto paths, auto seed, auto p, auto e) {
            qrh::launch_quadratic_rough_heston_asian_option_cuda<OptionSide::call,7U>(
                dm,1U,dd,1U,hp,dp,1U,PriceConstruction::CartesianProduct,
                1U,0U,1U,paths,1.0f/504.0f,2U,256U,1U,seed,p,e);
        });
}
