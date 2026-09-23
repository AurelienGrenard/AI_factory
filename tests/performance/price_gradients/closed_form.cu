// Public closed-form gradient geometry probe; each thread completes one row.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "tests/price_gradients/cuda_test_support.cuh"
#include "tests/performance/benchmark_support.cuh"

using namespace price_gradient_test;
namespace bs = model::equity::black_scholes;
int main(int argc,char** argv) {
    try {
        const unsigned threads=argc>1 ? std::stoul(argv[1]) : 256U;
        if (argc>2) throw std::invalid_argument("Usage: benchmark THREADS");
        for (std::size_t rows : {1U,1000U,65536U}) {
            std::vector<bs::ModelParameters> models;
            std::vector<product::EuropeanOptionParameters> products;
            for (std::size_t i=0;i<rows;++i) {
                models.push_back({.75f+.005f*float(i%101),.03f,.01f,.1f+.005f*float(i%61)});
                products.push_back({1.f,63U+std::uint32_t(i%190)});
            }
            const pg::PriceGradientConfiguration full{{{"model.spot",{.005}},
                {"model.volatility",{.005}}, {"product.strike",{.005}},
                {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
                {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
                {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
            for (std::size_t k=0;k<=6;++k) {
                auto selected=full; selected.sensitivities.resize(k);
                auto plan=bs::prepare_black_scholes_european_option_price_gradients(models,products,PriceConstruction::Aligned,{},selected);
                DeviceArray<bs::EuropeanOptionPriceGradientPlan::ScenarioType> inputs(plan.scenarios);
                DeviceArray<pg::Stencil> stencils(plan.stencils);
                DeviceArray<float> output(rows*(1+k));
                pg::DeviceInputs<bs::EuropeanOptionPriceGradientPlan::ScenarioType> device{inputs.data,inputs.count,stencils.data,stencils.count};
                pg::Outputs values{output.data,nullptr,k ? output.data+rows : nullptr,nullptr,rows,rows*k};
                pg::LaunchConfiguration config{pg::PricingMethod::closed_form,0,rows,0,threads,(rows+threads-1)/threads,0};
                auto launch=[&] {bs::launch_black_scholes_european_option_price_gradients_cuda<OptionSide::call>(plan,device,config,values);};
                auto timing=performance::measure_cuda(launch,5,21,64);
                auto host=output.read();
                std::uint64_t hash=14695981039346656037ULL;
                for (float v:host) { require(std::isfinite(v),"Nonfinite CF result"); hash^=std::bit_cast<std::uint32_t>(v); hash*=1099511628211ULL; }
                config.threads_per_block=256;config.block_count=(rows+255)/256;
                launch(); require(host==output.read(),"CF geometry changed output bits");
                std::cout << nlohmann::ordered_json{{"model","black_scholes"},{"method","closed_form"},
                    {"rows",rows},{"k",k},{"threads",threads},{"operations_per_sample",64},{"warmups",5},{"repetitions",21},
                    {"kernel",performance::timing_json(timing.kernel)},{"public_api",performance::timing_json(timing.wall)},
                    {"raw_host_clock",performance::timing_json(timing.raw_host_clock)},
                    {"environment",performance::environment_json()},{"output_hash",hash},{"matches_256",true}}.dump() << std::endl;
            }
        }
    } catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
