// Same-GPU comparison with the unchanged CEV price-delta public launcher.
#include "model/equity/markovian/cev/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/cev/product/european_option_price_delta.cuh"
#include "benchmark_support.cuh"
using namespace gradient_benchmark;
namespace cev=model::equity::cev;
int main() {
    try {
        const std::vector<cev::ModelParameters> models(64,{1.05f,.03f,.01f,.2f,.7f});
        const std::vector<product::EuropeanOptionParameters> products(64,{1.f,126U});
        const pg::PriceGradientConfiguration full{{{"model.spot",{.005}},
            {"model.sigma",{.005}}, {"model.beta",{.002,pg::BumpScale::absolute}},
            {"product.strike",{.005}}, {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
        DeviceBuffer<cev::ModelParameters> dm(models);
        DeviceBuffer<product::EuropeanOptionParameters> dp(products);
        for (unsigned k : {1U,4U,7U}) {
            auto selection=full;selection.sensitivities.resize(k);
            auto plan=cev::prepare_cev_european_option_price_gradients(models,products,PriceConstruction::Aligned,{},selection);
            if (k==1) measure("cev_legacy",plan,[&](const auto&,auto,const auto& config,auto out) {
                cev::launch_cev_european_option_price_delta_cuda<OptionSide::call>(models.data(),dm.data,64,
                    products.data(),dp.data,64,PriceConstruction::Aligned,64,0,64,config.paths_per_price,
                    1.f/504.f,2,256,64,config.base_seed,{.01f},out.prices,out.price_standard_errors,out.gradients,out.gradient_standard_errors);
            },256,1);
            measure("cev",plan,cev::launch_cev_european_option_price_gradients_cuda<OptionSide::call>,256,1);
        }
    } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
