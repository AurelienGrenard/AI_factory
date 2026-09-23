// Observe real transitions/payoff calls through the production terminal engine.
#include "common/philox.cuh"
#include "common/monte_carlo/price_gradients/kernel.cuh"
#include <iostream>

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace epg = equity::price_gradients;
namespace mcg = monte_carlo::price_gradients;
namespace {
__device__ unsigned int work_counts[3]; // all transitions, central transitions, central payoffs
struct Model { float spot, coefficient; };
struct Product { float strike; unsigned int endpoint; };
struct Dynamics {
    using ModelParameters = Model;
    using Prepared = Model;
    struct State { float value; };
    struct RandomContext {
        std::size_t path;
        __device__ RandomContext(philox::PhiloxKey, std::size_t p) : path(p) {}
    };
    static constexpr bool kExactTerminal = true;
    __device__ static Prepared prepare(Model model, float) { return model; }
    __device__ static State initial(Prepared model) { return {model.spot}; }
    __device__ static float draw(RandomContext& random) { return static_cast<float>(1U + random.path%3U); }
    __device__ static void transition(Prepared model, float innovation, const float*, State& state) {
        atomicAdd(&work_counts[0],1U);
        if (model.coefficient == 0.f) atomicAdd(&work_counts[1],1U);
        state.value += model.coefficient*innovation;
    }
    __device__ static float spot(State state) { return state.value; }
};
struct Payoff {
    using ProductParameters = Product;
    using PreparedProduct = Product;
    __device__ static Product prepare_product(Model, Product product, equity::ProductPreparationContext) { return product; }
    __device__ static int make_handler(Product) { return 0; }
    template<typename Observation>
    __device__ static float finalize(Product product, const typename Observation::State& state, int) {
        if (product.endpoint == 0U) atomicAdd(&work_counts[2],1U);
        return Observation::spot(state)-product.strike;
    }
    template<typename Observation>
    __device__ static float centered_difference(Product first, Product second,
        const typename Observation::State& a, const typename Observation::State& b, float width) {
        return (finalize<Observation>(second,b,0)-finalize<Observation>(first,a,0))/width;
    }
};
template<typename T> struct Buffer {
    T* data{};
    explicit Buffer(std::size_t n) { check_cuda(cudaMalloc(&data,n*sizeof(T)),"counter fixture allocate"); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void run(unsigned width, std::size_t blocks) {
    using Row = epg::Scenario<Model,Product>;
    constexpr std::size_t rows=2U, k=4U, paths=65U;
    const Row central{{1.f,0.f},{0.f,0U},1.f,1U,1.f,1.f,false,pg::CentralRequirement::payoff,{1.f,0.f,0.f}};
    std::vector<Row> scenarios;
    std::vector<pg::Stencil> stencils;
    for (std::size_t row=0; row<rows; ++row) {
        scenarios.push_back(central);
        for (unsigned parameter=0; parameter<k; ++parameter) {
            auto first=central, second=central;
            first.product.endpoint=1U; second.product.endpoint=2U;
            const bool shared = parameter==0U || parameter==2U;
            const bool boundary = parameter==3U;
            first.reuse_central=second.reuse_central=shared;
            first.central_requirement=second.central_requirement=boundary ? pg::CentralRequirement::payoff
                : shared ? pg::CentralRequirement::state : pg::CentralRequirement::none;
            const auto stencil=pg::prepare_stencil(0.f,{.125,pg::BumpScale::absolute},
                [&](float x){return !boundary || x>=0.f;});
            if (parameter==0U) { first.spot_scale=.875f; second.spot_scale=1.125f; }
            else if (parameter==2U) { first.product.strike=-.125f; second.product.strike=.125f; }
            else { first.model.coefficient=stencil.first; second.model.coefficient=stencil.second; }
            stencils.push_back(stencil);
            scenarios.push_back(first); scenarios.push_back(second);
        }
    }
    Buffer<Row> device_rows(scenarios.size()); Buffer<pg::Stencil> device_stencils(stencils.size());
    Buffer<float> values(2U*rows*(k+1U));
    check_cuda(cudaMemcpy(device_rows.data,scenarios.data(),scenarios.size()*sizeof(Row),cudaMemcpyHostToDevice),"upload rows");
    check_cuda(cudaMemcpy(device_stencils.data,stencils.data(),stencils.size()*sizeof(pg::Stencil),cudaMemcpyHostToDevice),"upload stencils");
    unsigned int counts[3]{};
    check_cuda(cudaMemcpyToSymbol(work_counts,counts,sizeof(counts)),"reset work counters");
    pg::LaunchConfiguration launch{pg::PricingMethod::monte_carlo,0U,rows,paths,64U,blocks,719U,width};
    mcg::launch<Dynamics,Payoff>({device_rows.data,scenarios.size(),device_stencils.data,stencils.size()},launch,{},
        {values.data,values.data+rows,values.data+2U*rows,values.data+2U*rows+rows*k,rows,rows*k},k,"central_work","fixture");
    check_cuda(cudaMemcpyFromSymbol(counts,work_counts,sizeof(counts)),"read work counters");
    const unsigned int expected[3]{width==1U?7U:width==2U?6U:5U, width==1U?3U:width==2U?2U:1U, width==4U?1U:2U};
    for (unsigned i=0;i<3U;++i) require(counts[i]==expected[i]*paths*rows,"Unexpected redundant/missing central work.");
    std::vector<float> output(2U*rows*(k+1U));
    check_cuda(cudaMemcpy(output.data(),values.data,output.size()*sizeof(float),cudaMemcpyDeviceToHost),"read counter results");
    double mean=0;
    for(std::size_t path=0;path<paths;++path) mean+=1U+path%3U;
    mean/=paths;
    for(std::size_t row=0;row<rows;++row) {
        require(output[row]==1.f && output[rows+row]==0.f,"Central output changed.");
        const float expected_gradient[4]{1.f,static_cast<float>(mean),-1.f,static_cast<float>(mean)};
        for(unsigned i=0;i<k;++i) require(output[2U*rows+row*k+i]==expected_gradient[i],"Synthetic gradient changed.");
    }
}
}  // namespace
int main() {
    try {
        int devices=0;
        if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        for(unsigned width:{1U,2U,4U}) { run(width,1U); run(width,2U*pg::sensitivity_batch_count(4U,width)); }
        std::cout << "Central work counts passed: no redundant centered payoff/trajectory; boundary and reuse retained\n";
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
