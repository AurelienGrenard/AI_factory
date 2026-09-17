// Diagnostic oracle: move only the deterministic carry/discount outside QE-M steps.
// Canonical central paths and Philox are unchanged; this is not a production pricer.
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "cuda_test_support.cuh"
#include <nlohmann/json.hpp>
using namespace price_gradient_test;
namespace hs=model::equity::heston;
using Scenario=hs::EuropeanOptionPriceGradientPlan::ScenarioType;
__global__ void rate_oracle(const Scenario* scenarios,double* gradients,std::size_t paths,float dt,
                            std::uint64_t seed) {
    const auto path=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (path>=paths) return;
    const auto row=std::size_t(blockIdx.y);const auto* s=scenarios+5*row;
    auto prepared=hs::prepare_model(s[0].model,dt);auto state=hs::initial_state(prepared);
    hs::DynamicsPolicy::RandomContext random(philox::make_key(seed+row),path);
    hs::DynamicsPolicy::advance(prepared,s[0].step_count,random,state);
    const double terminal=hs::DynamicsPolicy::spot(state);
    for (unsigned coordinate=0;coordinate<2;++coordinate) {
        const auto& a=s[1+2*coordinate];const auto& b=s[2+2*coordinate];
        const auto payoff=[&](const Scenario& endpoint) {
            const double carry=(double(endpoint.model.risk_free_rate)-s[0].model.risk_free_rate)
                -(double(endpoint.model.dividend_yield)-s[0].model.dividend_yield);
            return exp(-double(endpoint.model.risk_free_rate)*endpoint.maturity_years)
                * fmax(terminal*exp(carry*endpoint.maturity_years)-endpoint.product.strike,0.);
        };
        const double width=coordinate==0 ? double(b.model.risk_free_rate)-a.model.risk_free_rate
            : double(b.model.dividend_yield)-a.model.dividend_yield;
        gradients[(row*2+coordinate)*paths+path]=(payoff(b)-payoff(a))/width;
    }
}
int main() {
    try {
        constexpr std::size_t paths=262144;
        const std::vector<hs::ModelParameters> models{{1.05f,.03f,.01f,.04f,1.5f,.04f,.3f,-.7f},
            {1.f,0.f,0.f,.04f,1.f,.06f,.4f,-1.f}};
        const std::vector<product::EuropeanOptionParameters> products(2,{1.f,126U});
        for (unsigned refinement:{1U,2U}) for (double factor:{.125,.5,1.,2.,4.}) {
            auto plan=hs::prepare_heston_european_option_price_gradients(models,products,PriceConstruction::Aligned,
                {1.f/(504.f*refinement),2*refinement},{{{"model.risk_free_rate",{factor*.0001,pg::BumpScale::absolute}},
                {"model.dividend_yield",{factor*.0001,pg::BumpScale::absolute}}}});
            DeviceArray<Scenario> scenarios(plan.scenarios);DeviceArray<double> output(4*paths);
            for (unsigned seed:{719U,2719U,4719U}) {
                rate_oracle<<<dim3((paths+255)/256,2),256>>>(scenarios.data,output.data,paths,plan.time.dt,seed);
                check_cuda(cudaGetLastError(),"Heston carry oracle");auto values=output.read();
                for (unsigned i=0;i<4;++i) {
                    double sum=0,square=0;
                    for (std::size_t p=0;p<paths;++p) {auto v=values[i*paths+p];sum+=v;square+=v*v;}
                    const double mean=sum/paths,se=std::sqrt(std::max(0.,square/paths-mean*mean)/(paths-1));
                    std::cout<<nlohmann::ordered_json{{"row",i/2},{"coordinate",i%2==0 ? "model.risk_free_rate" : "model.dividend_yield"},
                        {"factor",factor},{"refinement",refinement},{"seed",seed},{"mean",mean},{"se",se}}.dump()<<std::endl;
                }
            }
        }
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
