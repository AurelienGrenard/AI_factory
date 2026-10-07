// Exercise dense barrier, discrete reset, and regular-observation rough graphs.
#include "common/check_cuda.cuh"
#include "common/volterra/hybrid_fft_workspace.cuh"
#include "model/equity/rough/rough_bergomi/product/down_and_out_option.cuh"
#include "model/equity/rough/rough_bergomi/product/down_and_out_option_price_gradients.cuh"
#include "model/equity/rough/rough_bergomi/product/forward_start_option.cuh"
#include "model/equity/rough/rough_bergomi/product/forward_start_option_price_gradients.cuh"
#include "model/equity/rough/rough_bergomi/product/cliquet.cuh"
#include "model/equity/rough/rough_bergomi/product/cliquet_price_gradients.cuh"
#include "model/equity/rough/rough_bergomi/product/athena_autocall.cuh"
#include "model/equity/rough/rough_bergomi/product/athena_autocall_price_gradients.cuh"
#include "model/equity/rough/rough_bergomi/product/geometric_asian_option.cuh"
#include "model/equity/rough/rough_bergomi/product/geometric_asian_option_price_gradients.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace rb = model::equity::rough_bergomi;
namespace pg = price_gradients;
namespace mcpg = monte_carlo::price_gradients;
void require(bool value, const char* why) {
    if (!value) throw std::runtime_error(why);
}
template<typename T> struct Buffer {
    T* ptr=nullptr;
    explicit Buffer(std::size_t n) { check_cuda(cudaMalloc(&ptr,n*sizeof(T)),"schedule graph alloc"); }
    ~Buffer(){cudaFree(ptr);}
    void upload(const T* data,std::size_t n){
        check_cuda(cudaMemcpy(ptr,data,n*sizeof(T),cudaMemcpyHostToDevice),"schedule graph upload");
    }
    void download(T* data,std::size_t n){
        check_cuda(cudaMemcpy(data,ptr,n*sizeof(T),cudaMemcpyDeviceToHost),"schedule graph download");
    }
};

// The node maturity changes the final event only. Capture the callbacks
// themselves so a shifted reset/coupon fails even if the price is noisy.
struct CalendarCapture {
    std::uint32_t* observed;
    std::uint32_t* weighted_payoff;

    __device__ bool on_initial_state(std::uint32_t) { return true; }
    __device__ bool on_observation(std::uint32_t index, std::uint32_t step) {
        observed[index] = step;
        *weighted_payoff += (index + 1U) * step;
        return true;
    }
};

template<typename Schedule>
__global__ void capture_calendar(
    typename Schedule::Calendar calendar,
    std::uint32_t steps,
    std::uint32_t* observed,
    std::uint32_t* weighted_payoff
) {
    const auto prepared = Schedule::prepare(
        calendar, {1.0f / 252.0f, 1.0f / 504.0f}, steps, 504U
    );
    auto cursor = Schedule::make_cursor(prepared);
    CalendarCapture capture{observed, weighted_payoff};
    Schedule::on_initial_state(prepared, cursor, 0U, capture);
    for (std::uint32_t step = 0U; step < steps; ++step)
        Schedule::on_step(prepared, cursor, step, step + 1U, capture);
}

template<typename Schedule>
void check_contractual_prefix(
    typename Schedule::Calendar calendar,
    const std::vector<std::uint32_t>& observation_days
) {
    const auto count = observation_days.size();
    Buffer<std::uint32_t> observed(count), weighted_payoff(1U);
    for (const auto steps : {503U, 504U, 505U}) {
        check_cuda(cudaMemset(observed.ptr, 0, count * sizeof(std::uint32_t)),
            "calendar observation reset");
        check_cuda(cudaMemset(weighted_payoff.ptr, 0, sizeof(std::uint32_t)),
            "calendar payoff reset");
        capture_calendar<Schedule><<<1U, 1U>>>(
            calendar, steps, observed.ptr, weighted_payoff.ptr
        );
        check_cuda(cudaDeviceSynchronize(), "calendar capture sync");
        std::vector<std::uint32_t> actual(count);
        std::uint32_t payoff = 0U;
        observed.download(actual.data(), count);
        weighted_payoff.download(&payoff, 1U);
        std::uint32_t expected_payoff = 0U;
        for (std::size_t index = 0U; index < count; ++index) {
            const auto expected = index + 1U == count
                ? steps : 2U * observation_days[index];
            require(actual[index] == expected,
                "Maturity bump moved a contractual prefix observation.");
            expected_payoff += static_cast<std::uint32_t>(index + 1U)
                * expected;
        }
        require(payoff == expected_payoff,
            "Maturity bump changed the pathwise calendar payoff.");
    }
}

void check_maturity_calendar_nodes() {
    std::vector<std::uint32_t> regular_days;
    for (std::uint32_t observation = 1U; observation <= 12U; ++observation)
        regular_days.push_back(21U * observation);
    check_contractual_prefix<volterra::RegularHybridSchedule>(
        simulation::RegularCalendar{21U, 12U}, regular_days
    );
    std::vector<std::uint32_t> stubbed_days;
    for (std::uint32_t observation = 0U; observation < 11U; ++observation)
        stubbed_days.push_back(42U + 21U * observation);
    check_contractual_prefix<volterra::StubbedRegularHybridSchedule>(
        simulation::StubbedRegularCalendar{42U, 21U, 11U}, stubbed_days
    );
    check_contractual_prefix<volterra::CalendarHybridSchedule<2U>>(
        simulation::StaticCalendar<2U>{{126U, 126U}}, {126U, 252U}
    );
}

template<typename Graph,typename Product,typename Prepare,typename Reference>
void run(const Product& product,Prepare prepare,Reference reference) {
    constexpr std::size_t paths=257U;
    constexpr std::uint64_t seed=867124U;
    const std::vector<rb::ModelParameters> models{{
        1.0f,.02f,.01f,.04f,.8f,.10f,-.7f
    }};
    const std::vector<Product> products{product};
    const pg::PriceGradientConfiguration bump{{
        {"model.spot",{.01f,pg::BumpScale::relative}}
    }};
    const auto host=prepare(models,products,PriceConstruction::CartesianProduct,
        pg::TimeConfiguration{1.0f/504.0f,2U},bump,
        pg::SensitivityRequest{pg::SensitivityOrders::first_and_second});
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo,0U,1U,paths,256U,1U,seed
    };
    typename Graph::Cache cache;
    const auto plan=Graph::plan(host,cache,launch);
    Buffer<unsigned char> workspace(plan.bytes);
    Buffer<float> prices(1),errors(1),grads(1),grad_errors(1),
        hessians(1),hessian_errors(1);
    Buffer<pg::SensitivityStencil<4U>> stencils(1);
    Graph::launch(host,cache,launch,
        {prices.ptr,errors.ptr,grads.ptr,grad_errors.ptr,
         hessians.ptr,hessian_errors.ptr,1U,1U},
        mcpg::DevicePreparedStencilOutputs<4U>{stencils.ptr,1U,nullptr},
        plan,workspace.ptr,plan.bytes);
    check_cuda(cudaDeviceSynchronize(),"schedule graph sync");
    float graph_price=0.0f,gradient=0.0f,hessian=0.0f;
    prices.download(&graph_price,1);
    grads.download(&gradient,1);
    hessians.download(&hessian,1);
    require(std::isfinite(graph_price)&&std::isfinite(gradient)
        &&std::isfinite(hessian),"Invalid path schedule graph result.");
    const auto steps=2U*product.maturity_days;
    const auto bytes=volterra::required_hybrid_fft_workspace_bytes(
        steps,paths,256U);
    Buffer<unsigned char> reference_workspace(bytes);
    Buffer<rb::ModelParameters> dm(1);
    Buffer<Product> dp(1);
    Buffer<float> reference_price(1),reference_error(1);
    dm.upload(models.data(),1);
    dp.upload(products.data(),1);
    reference(dm.ptr,dp.ptr,steps,paths,
        reference_workspace.ptr,bytes,seed,
        reference_price.ptr,reference_error.ptr);
    check_cuda(cudaDeviceSynchronize(),"schedule reference sync");
    float expected=0.0f;
    reference_price.download(&expected,1);
    require(std::abs(graph_price-expected)<5.0e-5f,
        "Path schedule graph price differs from FFT pricer.");
    float endpoint_prices[2]{};
    for (std::size_t endpoint=0U; endpoint<2U; ++endpoint) {
        auto bumped=models[0U];
        bumped.spot *= endpoint==0U ? .99f : 1.01f;
        dm.upload(&bumped,1);
        reference(dm.ptr,dp.ptr,steps,paths,
            reference_workspace.ptr,bytes,seed,
            reference_price.ptr,reference_error.ptr);
        check_cuda(cudaDeviceSynchronize(),"schedule CRN reference sync");
        reference_price.download(&endpoint_prices[endpoint],1);
    }
    const float oracle_delta=(endpoint_prices[1]-endpoint_prices[0])
        / (.02f*models[0U].spot);
    require(std::abs(gradient-oracle_delta)<5.0e-3f,
        "Path schedule graph spot delta differs from paired FFT prices.");
}

template<typename Graph, typename Product, typename Prepare>
void run_maturity_graph(const Product& product, Prepare prepare) {
    constexpr std::size_t paths = 257U;
    const std::vector<rb::ModelParameters> models{{
        1.0f, .02f, .01f, .04f, .8f, .10f, -.7f
    }};
    const std::vector<Product> products{product};
    const pg::PriceGradientConfiguration bump{{
        {"product.maturity_years", {1.0f / 504.0f,
                                     pg::BumpScale::absolute}}
    }};
    const auto host = prepare(
        models, products, PriceConstruction::CartesianProduct,
        pg::TimeConfiguration{1.0f / 504.0f, 2U}, bump,
        pg::SensitivityRequest{pg::SensitivityOrders::first_and_second}
    );
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, 1U, paths, 256U, 1U, 867124U
    };
    typename Graph::Cache cache;
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
    check_cuda(cudaDeviceSynchronize(), "maturity path graph sync");
    float price = 0.0f, gradient = 0.0f, hessian = 0.0f;
    prices.download(&price, 1U);
    gradients.download(&gradient, 1U);
    hessians.download(&hessian, 1U);
    require(std::isfinite(price) && std::isfinite(gradient)
        && std::isfinite(hessian),
        "Invalid path graph maturity node.");
}
}
int main() {
    int count=0;
    const auto availability=cudaGetDeviceCount(&count);
    if (availability==cudaErrorNoDevice
        || availability==cudaErrorInsufficientDriver || count==0) return 77;
    check_cuda(availability,"schedule graph cudaGetDeviceCount");
    check_maturity_calendar_nodes();
    run_maturity_graph<rb::ForwardStartOptionNodeGraph<OptionSide::call,
            pg::SensitivityOrders::first_and_second, 1U>>(
        product::ForwardStartOptionParameters{1.0f, 126U, 252U},
        rb::prepare_rough_bergomi_forward_start_option_sensitivities
    );
    run_maturity_graph<rb::CliquetNodeGraph<
            pg::SensitivityOrders::first_and_second, 1U>>(
        product::CliquetParameters{252U, 21U, 1.0f, -.1f, .1f, -.2f, .2f},
        rb::prepare_rough_bergomi_cliquet_sensitivities
    );
    run<rb::DownAndOutOptionNodeGraph<OptionSide::call,
            pg::SensitivityOrders::first_and_second,1U>>(
        product::DownAndOutOptionParameters{1.0f,.75f,63U},
        rb::prepare_rough_bergomi_down_and_out_option_sensitivities,
        [](auto dm,auto dp,auto steps,auto paths,auto ws,auto bytes,
           auto seed,auto price,auto error){
            rb::launch_rough_bergomi_down_and_out_option_cuda<OptionSide::call>(
                dm,1U,dp,1U,PriceConstruction::CartesianProduct,
                1U,0U,paths,1.0f/252.0f,1.0f/504.0f,steps,256U,
                ws,bytes,seed,price,error);
        });
    run<rb::ForwardStartOptionNodeGraph<OptionSide::call,
            pg::SensitivityOrders::first_and_second,1U>>(
        product::ForwardStartOptionParameters{1.0f,21U,63U},
        rb::prepare_rough_bergomi_forward_start_option_sensitivities,
        [](auto dm,auto dp,auto steps,auto paths,auto ws,auto bytes,
           auto seed,auto price,auto error){
            rb::launch_rough_bergomi_forward_start_option_cuda<OptionSide::call>(
                dm,1U,dp,1U,PriceConstruction::CartesianProduct,
                1U,0U,paths,1.0f/252.0f,1.0f/504.0f,steps,256U,
                ws,bytes,seed,price,error);
        });
    run<rb::CliquetNodeGraph<pg::SensitivityOrders::first_and_second,1U>>(
        product::CliquetParameters{63U,21U,1.0f,-.1f,.1f,-.2f,.2f},
        rb::prepare_rough_bergomi_cliquet_sensitivities,
        [](auto dm,auto dp,auto steps,auto paths,auto ws,auto bytes,
           auto seed,auto price,auto error){
            rb::launch_rough_bergomi_cliquet_cuda(
                dm,1U,dp,1U,PriceConstruction::CartesianProduct,
                1U,0U,paths,1.0f/252.0f,1.0f/504.0f,steps,256U,
                ws,bytes,seed,price,error);
        });
    run<rb::AthenaAutocallNodeGraph<
            pg::SensitivityOrders::first_and_second,1U>>(
        product::AthenaAutocallParameters{63U,21U,1.01f,.75f,.10f},
        rb::prepare_rough_bergomi_athena_autocall_sensitivities,
        [](auto dm,auto dp,auto steps,auto paths,auto ws,auto bytes,
           auto seed,auto price,auto error){
            rb::launch_rough_bergomi_athena_autocall_cuda(
                dm,1U,dp,1U,PriceConstruction::CartesianProduct,
                1U,0U,paths,1.0f/252.0f,1.0f/504.0f,steps,256U,
                ws,bytes,seed,price,error);
        });
    run<rb::GeometricAsianOptionNodeGraph<OptionSide::call,
            pg::SensitivityOrders::first_and_second,1U>>(
        product::GeometricAsianOptionParameters{1.0f,63U},
        rb::prepare_rough_bergomi_geometric_asian_option_sensitivities,
        [](auto dm,auto dp,auto steps,auto paths,auto ws,auto bytes,
           auto seed,auto price,auto error){
            rb::launch_rough_bergomi_geometric_asian_option_cuda<
                OptionSide::call>(
                dm,1U,dp,1U,PriceConstruction::CartesianProduct,
                1U,0U,paths,1.0f/252.0f,1.0f/504.0f,steps,256U,
                ws,bytes,seed,price,error);
        });
}
