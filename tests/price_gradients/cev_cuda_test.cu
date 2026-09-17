// Nonhomogeneous paths, absorbing CEV states, boundaries and legacy delta parity.
#include "model/equity/markovian/cev/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/cev/product/european_option_price_delta.cuh"
#include "cuda_test_support.cuh"

using namespace price_gradient_test;
namespace cev = model::equity::cev;

template<OptionSide Side> void check(std::size_t paths) {
    const std::vector<cev::ModelParameters> models{
        {1.05f,.03f,.01f,.2f,.7f}, {1.f,0.f,0.f,.3f,.999f}, {.01f,0.f,0.f,2.f,.5f}};
    const std::vector<product::EuropeanOptionParameters> products{{1.f,16U},{1.f,8U},{.02f,4U}};
    const pg::Sensitivity spot{"model.spot",{.005f}};
    const pg::PriceGradientConfiguration full{{spot,
        {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
        {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
        {"model.sigma",{.005}}, {"model.beta",{.002,pg::BumpScale::absolute}},
        {"product.strike",{.005}}, {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
    auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return cev::prepare_cev_european_option_price_gradients(models,products,PriceConstruction::Aligned,{},selection);
    };
    auto launcher=cev::launch_cev_european_option_price_gradients_cuda<Side>;
    auto plan=prepare(full);
    require(!plan.scenarios[1].reuse_central && plan.scenarios[1].spot_scale==1.f
        && plan.scenarios[1].simulation_spot==plan.scenarios[1].model.spot,
        "CEV spot bump incorrectly uses multiplicative reuse.");
    require(plan.stencils[7+4].kind==pg::StencilKind::backward
        && plan.stencils[14+4].kind==pg::StencilKind::forward,"CEV beta boundaries not resolved.");
    DeviceArray<cev::ModelParameters> dm(models);
    DeviceArray<product::EuropeanOptionParameters> dp(products);
    for (unsigned threads : {64U,128U,256U}) {
        pg::LaunchConfiguration launch{pg::PricingMethod::monte_carlo,0U,3U,paths,threads,3U,719U};
        const auto solo=execute(prepare({{spot}}),launch,launcher);
        DeviceArray<float> old(12);
        cev::launch_cev_european_option_price_delta_cuda<Side>(models.data(),dm.data,3U,
            products.data(),dp.data,3U,PriceConstruction::Aligned,3U,0U,3U,paths,
            1.f/504.f,2U,threads,3U,719U,{.01f},old.data,old.data+3,old.data+6,old.data+9);
        auto reference=old.read();
        for (unsigned row=0;row<3;++row) {
            same(solo.price[row],reference[row],"CEV legacy price");
            same(solo.price_error[row],reference[3+row],"CEV legacy price SE");
            same(solo.gradient[row],reference[6+row],"CEV legacy delta");
            same(solo.gradient_error[row],reference[9+row],"CEV legacy delta SE");
        }
        const auto all=execute(plan,launch,launcher);
        batching(full,all,launch,prepare,launcher);
        for (std::size_t i=0;i<7;++i) {
            auto one=execute(prepare({{full.sensitivities[i]}}),launch,launcher);
            for (std::size_t row=0;row<3;++row) {
                same(all.gradient[row*7+i],one.gradient[row],"CEV selected coordinate");
                same(all.gradient_error[row*7+i],one.gradient_error[row],"CEV selected SE");
            }
        }
        const auto reordered=execute(prepare({{full.sensitivities[4],spot,full.sensitivities[5]}}),launch,launcher);
        for (std::size_t row=0;row<3;++row)
            same(reordered.gradient[row*3+1],solo.gradient[row],"CEV selection order");
    }
    // Cartesian decoding and global row seeds must equal explicit aligned expansion.
    std::vector<cev::ModelParameters> expanded_models;
    std::vector<product::EuropeanOptionParameters> expanded_products;
    for (auto m:models) for (auto p:products) { expanded_models.push_back(m); expanded_products.push_back(p); }
    auto cart=cev::prepare_cev_european_option_price_gradients(models,products,PriceConstruction::CartesianProduct,{},full);
    auto aligned=cev::prepare_cev_european_option_price_gradients(expanded_models,expanded_products,PriceConstruction::Aligned,{},full);
    pg::LaunchConfiguration config{pg::PricingMethod::monte_carlo,0,9,paths,256,9,719};
    auto a=execute(cart,config,launcher,true), b=execute(aligned,config,launcher);
    require(a.price==b.price && a.gradient==b.gradient && a.gradient_error==b.gradient_error,
        "CEV Cartesian/split row mapping changed outputs.");
    std::cout << "CEV " << option_side_name(Side) << ": seven coordinates, legacy parity, absorption, boundaries, all batches and Cartesian rows passed\n";
}
int main(int argc,char** argv) {
    try {
        std::size_t paths=4097;
        if (argc==2 && std::string_view(argv[1])=="--sanitizer") paths=513;
        else if (argc!=1) throw std::invalid_argument("Usage: test [--sanitizer]");
        int devices=0;
        if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        check<OptionSide::call>(paths); check<OptionSide::put>(paths);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
