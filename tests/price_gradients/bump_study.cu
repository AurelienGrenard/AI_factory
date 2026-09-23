// Opt-in numerical study: represented stencils and CRN estimates, without publication.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/cev/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "cuda_test_support.cuh"
#include <nlohmann/json.hpp>
using namespace price_gradient_test;
namespace bs=model::equity::black_scholes;
namespace cv=model::equity::cev;
namespace hs=model::equity::heston;
using Json=nlohmann::ordered_json;

template<typename ModelPolicy,typename Plan>
Json scenario_json(const Plan& plan) {
    Json result=Json::array();
    for (auto& s:plan.scenarios) {
        Json m;
        for (auto& f:ModelPolicy::fields) m[std::string(f.name)]=s.model.*f.member;
        result.push_back({{"model",m},{"strike",s.product.strike},{"maturity",s.maturity_years},{"steps",s.step_count}});
    }
    return result;
}
template<typename ModelPolicy,typename Prepare,typename Launcher>
void study(const char* model,const pg::PriceGradientConfiguration& full,Prepare prepare,Launcher launcher,std::size_t paths,
           bool rates_only,const std::vector<std::string>& case_ids,
           const std::vector<std::vector<std::string>>& case_classes) {
    const std::vector<double> factors=rates_only ? std::vector<double>{1.,2.,5.,10.}
        : std::vector<double>{.125,.5,1.,2.,4.,5.,10.};
    for (unsigned refinement : {1U,2U}) {
        if (std::string_view(model)=="black_scholes" && refinement==2U) continue;
        for (double factor : factors) {
            auto selection=full;
            if (rates_only) selection.sensitivities={pg::Sensitivity{"model.risk_free_rate",{.0001,pg::BumpScale::absolute}}};
            for (auto& c:selection.sensitivities) {
                if (c.parameter=="product.maturity_years")
                    c.bump.displacement=float(std::max(1.,factor*2.))*(1.f/504.f);
                else c.bump.displacement*=factor;
            }
            auto plan=prepare(selection,refinement);
            if (plan.result_count!=case_ids.size() || case_ids.size()!=case_classes.size())
                throw std::runtime_error("qualification case metadata does not match prepared rows");
            Json stencils=Json::array();
            for (auto& s:plan.stencils) stencils.push_back({{"central",s.central},{"first",s.first},{"second",s.second},
                {"width",s.represented_width},{"w1",s.first_weight},{"w2",s.second_weight},{"kind",int(s.kind)}});
            for (unsigned seed : {719U,2719U,4719U}) {
                pg::LaunchConfiguration config{pg::PricingMethod::monte_carlo,0,plan.result_count,paths,256,plan.result_count,seed};
                auto result=execute(plan,config,launcher);
                Json report{{"model",model},{"factor",factor},{"refinement",refinement},{"seed",seed},{"paths",paths},
                    {"schema_version",2},{"threads_per_block",config.threads_per_block},
                    {"sensitivity_batch_size",config.sensitivity_batch_size},
                    {"dt",plan.time.dt},{"rows",plan.result_count},{"k",plan.sensitivity_count()},
                    {"scenarios",scenario_json<ModelPolicy>(plan)},{"stencils",stencils},
                    {"price",result.price},{"price_se",result.price_error},{"gradient",result.gradient},{"gradient_se",result.gradient_error}};
                report["study"]=rates_only ? "rate_basis_points" : "all_coordinates";
                report["policy_id"]="equity-european-price-gradient-bumps-v2";
                report["case_ids"]=case_ids;
                report["case_classes"]=case_classes;
                if (rates_only) report["requested_bump_basis_points"]=factor;
                report["coordinates"]=Json::array();
                for (auto& s:selection.sensitivities) report["coordinates"].push_back(s.parameter);
                if (std::string_view(model)=="black_scholes") {
                    config.method=pg::PricingMethod::closed_form;config.block_count=1;
                    auto exact=execute(plan,config,launcher);report["cf_price"]=exact.price;report["cf_gradient"]=exact.gradient;
                }
                std::cout << report.dump() << std::endl;
            }
        }
    }
}
int main(int argc,char** argv) {
    try {
        const std::size_t paths=argc>1 ? std::stoull(argv[1]) : 262144U;
        const bool rates_only=argc==3 && std::string_view(argv[2])=="--rates";
        if (argc>3 || (argc==3 && !rates_only) || paths<2)
            throw std::invalid_argument("Usage: bump_study [PATHS [--rates]]");
        const pg::PriceGradientConfiguration bc{{{"model.spot",{.005}}, {"model.volatility",{.005}},
            {"product.strike",{.005}}, {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
        std::vector<bs::ModelParameters> bm{{1.05f,.03f,.01f,.2f},{1.f,0.f,0.f,.2f},{1.f,0.f,0.f,.005f}};
        std::vector<product::EuropeanOptionParameters> bp{{1.f,126U},{1.f,1U},{1.f,126U}};
        bm.insert(bm.end(),{{1.05f,-.01f,.01f,.2f},{1.05f,.10f,.01f,.2f},{105.f,.03f,.01f,.2f}});
        bp.insert(bp.end(),{{1.f,126U},{1.f,504U},{100.f,126U}});
        const std::vector<std::string> bs_ids{"ordinary","one_day","low_volatility",
            "negative_rate","high_rate_long_maturity","large_notional_scale"};
        const std::vector<std::vector<std::string>> bs_classes{{"ordinary"},{"short_maturity","boundary"},
            {"stress","boundary"},{"stress"},{"stress","maturity_scale"},{"scale"}};
        study<bs::price_gradients::ParameterPolicy>("black_scholes",bc,[&](auto& c,unsigned){
            return bs::prepare_black_scholes_european_option_price_gradients(bm,bp,PriceConstruction::Aligned,{},c);
        },bs::launch_black_scholes_european_option_price_gradients_cuda<OptionSide::call>,paths,rates_only,
            bs_ids,bs_classes);
        const pg::PriceGradientConfiguration hc{{{"model.spot",{.005}},
            {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}}, {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"model.initial_variance",{.001,pg::BumpScale::absolute}}, {"model.kappa",{.005}},
            {"model.theta",{.005}}, {"model.gamma",{.005}}, {"model.rho",{.002,pg::BumpScale::absolute}},
            {"product.strike",{.005}}, {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
        std::vector<hs::ModelParameters> hm{{1.05f,.03f,.01f,.04f,1.5f,.04f,.3f,-.7f},{1.f,0.f,0.f,.04f,1.f,.06f,.4f,-1.f},
            {1.f,0.f,0.f,.04f,1.f,.06f,.4f,-.7f}};
        std::vector<product::EuropeanOptionParameters> hp{{1.f,126U},{1.f,126U},{1.f,1U}};
        hm.insert(hm.end(),{{1.05f,-.01f,.01f,.04f,1.5f,.04f,.3f,-.7f},
            {1.05f,.10f,.01f,.04f,1.5f,.04f,.3f,-.7f},{105.f,.03f,.01f,.04f,1.5f,.04f,.3f,-.7f}});
        hp.insert(hp.end(),{{1.f,126U},{1.f,504U},{100.f,126U}});
        const std::vector<std::string> heston_ids{"ordinary","rho_lower_boundary","one_day","negative_rate",
            "high_rate_long_maturity","large_notional_scale"};
        const std::vector<std::vector<std::string>> heston_classes{{"ordinary"},{"stress","boundary"},
            {"short_maturity","boundary"},{"stress"},{"stress","maturity_scale"},{"scale"}};
        study<hs::price_gradients::ParameterPolicy>("heston",hc,[&](auto& c,unsigned refinement){
            return hs::prepare_heston_european_option_price_gradients(hm,hp,PriceConstruction::Aligned,
                {1.f/(504.f*refinement),2U*refinement},c);
        },hs::launch_heston_european_option_price_gradients_cuda<OptionSide::call>,paths,rates_only,
            heston_ids,heston_classes);
        if (!rates_only) {
            const pg::PriceGradientConfiguration cc{{{"model.spot",{.005}},
                {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
                {"model.dividend_yield",{.0001,pg::BumpScale::absolute}}, {"model.sigma",{.005}},
                {"model.beta",{.002,pg::BumpScale::absolute}}, {"product.strike",{.005}},
                {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
            const std::vector<cv::ModelParameters> cm{{1.05f,.03f,.01f,.25f,.75f},
                {1.f,0.f,0.f,.2f,.5f},{1.f,0.f,0.f,.2f,.995f},{1.f,0.f,0.f,.2f,.75f},
                {1.f,0.f,0.f,.01f,.75f},{1.05f,-.01f,.01f,.25f,.75f},
                {1.05f,.10f,.01f,.25f,.75f},{105.f,.03f,.01f,2.5f,.75f}};
            const std::vector<product::EuropeanOptionParameters> cp{{1.f,126U},{1.f,126U},{1.f,126U},
                {1.f,1U},{1.f,126U},{1.f,126U},{1.f,504U},{100.f,126U}};
            const std::vector<std::string> cev_ids{"ordinary","beta_lower_boundary",
                "beta_upper_near_boundary","one_day","low_sigma","negative_rate",
                "high_rate_long_maturity","large_notional_scale"};
            const std::vector<std::vector<std::string>> cev_classes{{"ordinary"},{"boundary"},
                {"stress","boundary"},{"short_maturity","boundary"},{"stress","boundary"},
                {"stress"},{"stress","maturity_scale"},{"scale"}};
            study<cv::price_gradients::ParameterPolicy>("cev",cc,[&](auto& c,unsigned refinement){
                return cv::prepare_cev_european_option_price_gradients(cm,cp,PriceConstruction::Aligned,
                    {1.f/(504.f*refinement),2U*refinement},c);
            },cv::launch_cev_european_option_price_gradients_cuda<OptionSide::call>,paths,false,
                cev_ids,cev_classes);
        }
    } catch (const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
