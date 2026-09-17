// Fixed-horizon Merton CRN coupling and legacy spot-delta compatibility.
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_delta.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"
#include "cuda_test_support.cuh"
#include <algorithm>
#include <cmath>

using namespace price_gradient_test;
namespace merton = model::equity::merton;
namespace philox = ai_factory::workbench::philox;

__global__ void legacy_factorization_kernel(merton::ModelParameters parameters,float maturity,
                                             std::uint64_t seed,float* output,std::size_t paths) {
    for (std::size_t path=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;path<paths;
         path+=std::size_t(gridDim.x)*blockDim.x) {
        const auto prepared_model=merton::prepare_model(parameters);
        const auto prepared_transition=merton::prepare_transition(prepared_model,maturity);
        auto state=merton::initial_state(prepared_model);
        philox::NormalRandomContext random(philox::make_key(seed),path);
        constexpr float threshold=10.f;
        const std::uint32_t count=prepared_transition.poisson_mean<threshold
            ? philox::poisson_from_uniform(random.uniforms.next(),prepared_transition.poisson_mean,
                                           prepared_transition.zero_jump_probability)
            : philox::poisson_from_uniform_sequence(random.uniforms,prepared_transition.poisson_mean);
        const float diffusion=philox::next_normal(random.uniforms,random.normals);
        const float jump=count==0U ? 0.f : philox::next_normal(random.uniforms,random.normals);
        merton::one_step_transition(prepared_model,prepared_transition,count,diffusion,jump,state);
        output[path]=state.log_spot;
    }
}

__global__ void canonical_factorization_kernel(merton::ModelParameters parameters,float maturity,
                                                std::uint64_t seed,float* output,std::size_t paths) {
    for (std::size_t path=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;path<paths;
         path+=std::size_t(gridDim.x)*blockDim.x) {
        const auto prepared_model=merton::prepare_model(parameters);
        const auto prepared_transition=merton::prepare_transition(prepared_model,maturity);
        auto state=merton::initial_state(prepared_model);
        philox::NormalRandomContext random(philox::make_key(seed),path);
        const auto innovations=merton::draw_transition_innovations(prepared_transition,random);
        merton::one_step_transition(prepared_model,prepared_transition,innovations.jump_count,
                                    innovations.diffusion_normal,innovations.jump_normal,state);
        output[path]=state.log_spot;
    }
}

void factorization_benchmark() {
    constexpr std::size_t paths=1U<<20U;
    constexpr unsigned threads=256U, iterations=20U, rounds=10U;
    const unsigned blocks=unsigned((paths+threads-1U)/threads);
    const merton::ModelParameters parameters{1.05f,.03f,.01f,.2f,25.f,-.1f,.25f};
    DeviceArray<float> legacy(paths),canonical(paths);
    legacy_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U,legacy.data,paths);
    canonical_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U,canonical.data,paths);
    check_cuda(cudaDeviceSynchronize(),"Merton factorization warmup");
    const auto before=legacy.read(),after=canonical.read();
    for (std::size_t i=0;i<paths;++i) same(before[i],after[i],"Merton canonical draw changed a terminal bit");
    cudaEvent_t start{},stop{};check_cuda(cudaEventCreate(&start),"factorization event");
    check_cuda(cudaEventCreate(&stop),"factorization event");
    auto elapsed=[&](bool use_legacy) {
        check_cuda(cudaEventRecord(start),"factorization start");
        for (unsigned i=0;i<iterations;++i) {
            if (use_legacy) legacy_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U+i,legacy.data,paths);
            else canonical_factorization_kernel<<<blocks,threads>>>(parameters,2.f,719U+i,canonical.data,paths);
        }
        check_cuda(cudaEventRecord(stop),"factorization stop");check_cuda(cudaEventSynchronize(stop),"factorization sync");
        float milliseconds=0;check_cuda(cudaEventElapsedTime(&milliseconds,start,stop),"factorization time");
        return milliseconds/iterations;
    };
    float legacy_ms=0.f,canonical_ms=0.f;
    for (unsigned round=0;round<rounds;++round) {
        if ((round&1U)==0U) { legacy_ms+=elapsed(true);canonical_ms+=elapsed(false); }
        else { canonical_ms+=elapsed(false);legacy_ms+=elapsed(true); }
    }
    legacy_ms/=rounds;canonical_ms/=rounds;
    cudaFuncAttributes legacy_attributes{},canonical_attributes{};
    check_cuda(cudaFuncGetAttributes(&legacy_attributes,legacy_factorization_kernel),"legacy attributes");
    check_cuda(cudaFuncGetAttributes(&canonical_attributes,canonical_factorization_kernel),"canonical attributes");
    cudaEventDestroy(start);cudaEventDestroy(stop);
    std::cout << "{\"paths\":" << paths << ",\"iterations_per_round\":" << iterations
              << ",\"rounds\":" << rounds
              << ",\"legacy_ms\":" << legacy_ms << ",\"canonical_ms\":" << canonical_ms
              << ",\"legacy_registers\":" << legacy_attributes.numRegs
              << ",\"canonical_registers\":" << canonical_attributes.numRegs
              << ",\"legacy_local_bytes\":" << legacy_attributes.localSizeBytes
              << ",\"canonical_local_bytes\":" << canonical_attributes.localSizeBytes
              << ",\"bitwise_equal\":true}\n";
}

double normal_cdf(double value) { return .5 * std::erfc(-value/std::sqrt(2.)); }

template<OptionSide Side,typename Scenario>
double independent_price(const Scenario& scenario) {
    const auto& model=scenario.model;
    const double t=scenario.maturity_years, strike=scenario.product.strike;
    const double jump_variance=double(model.jump_log_volatility)*model.jump_log_volatility;
    const double compensator=std::exp(double(model.jump_log_mean)+.5*jump_variance)-1.;
    const double poisson_mean=double(model.jump_intensity)*t;
    double weight=std::exp(-poisson_mean), total=0., probability=0.;
    for (unsigned count=0;count<1024;++count) {
        const double variance=double(model.volatility)*model.volatility*t+count*jump_variance;
        const double mean=std::log(double(model.spot))
            +(double(model.risk_free_rate)-model.dividend_yield-model.jump_intensity*compensator
              -.5*double(model.volatility)*model.volatility)*t+count*model.jump_log_mean;
        const double root=std::sqrt(variance);
        double call;
        if (root==0.) call=std::max(std::exp(mean)-strike,0.);
        else {
            const double d2=(mean-std::log(strike))/root, d1=d2+root;
            call=std::exp(mean+.5*variance)*normal_cdf(d1)-strike*normal_cdf(d2);
        }
        const double discount=std::exp(-double(model.risk_free_rate)*t);
        const double conditional = Side==OptionSide::call ? discount*call
            : discount*(call-std::exp(mean+.5*variance)+strike);
        total+=weight*conditional; probability+=weight;
        if (count>poisson_mean && 1.-probability<1e-14) break;
        weight*=poisson_mean/double(count+1);
    }
    return total;
}

template<OptionSide Side,typename Plan>
std::pair<double,double> independent_gradients(const Plan& plan,const Results& result) {
    const auto k=plan.sensitivity_count(), n=1U+2U*k;
    double maximum_absolute=0.,maximum_budget_fraction=0.;
    for (std::size_t row=0;row<plan.result_count;++row) {
        const double central=independent_price<Side>(plan.scenarios[row*n]);
        for (std::size_t i=0;i<k;++i) {
            const double first=independent_price<Side>(plan.scenarios[row*n+2*i+1]);
            const double second=independent_price<Side>(plan.scenarios[row*n+2*i+2]);
            const auto& stencil=plan.stencils[row*k+i];
            const double reference=stencil.kind==pg::StencilKind::centered
                ? (second-first)/stencil.represented_width
                : stencil.first_weight*(first-central)+stencil.second_weight*(second-central);
            const double estimate=result.gradient[row*k+i], se=result.gradient_error[row*k+i];
            const double numerical=5e-4+5e-3*std::abs(reference);
            const double error=std::abs(estimate-reference),budget=6.*se+numerical;
            maximum_absolute=std::max(maximum_absolute,error);
            maximum_budget_fraction=std::max(maximum_budget_fraction,error/budget);
            if (error>budget) {
                std::cerr << "Merton independent gradient mismatch row=" << row << " coordinate=" << i
                          << " estimate=" << estimate << " reference=" << reference << " se=" << se << '\n';
                throw std::runtime_error("Merton independent represented-stencil comparison failed");
            }
            if (model::equity::merton::price_gradients::ParameterPolicy::fields.size()>i
                && model::equity::merton::price_gradients::ParameterPolicy::fields[i].name.starts_with("model.jump_")
                && plan.scenarios[row*n].model.jump_intensity==0.f) {
                if (!(std::abs(estimate)<1e-5 && std::abs(reference)<1e-9)) {
                    std::cerr << "Zero-intensity jump gradient estimate=" << estimate
                              << " reference=" << reference << " coordinate=" << i << '\n';
                    throw std::runtime_error("Zero-intensity Merton jump gradient is not zero");
                }
            }
        }
    }
    return {maximum_absolute,maximum_budget_fraction};
}

template<OptionSide Side> void check(std::size_t paths) {
    const std::vector<merton::ModelParameters> models{
        {1.05f,.03f,.01f,.2f,.5f,-.1f,.25f},
        {1.f,-.01f,0.f,.35f,25.f,.02f,.4f},
        {100.f,.08f,.02f,.1f,0.f,-.2f,.1f}};
    const std::vector<product::EuropeanOptionParameters> products{{1.f,126U},{1.1f,126U},{95.f,504U}};
    const pg::Sensitivity spot{"model.spot",{.005f}};
    const pg::PriceGradientConfiguration full{{spot,
        {"model.risk_free_rate",{.0005,pg::BumpScale::absolute}},
        {"model.dividend_yield",{.0005,pg::BumpScale::absolute}}, {"model.volatility",{.005}},
        {"model.jump_log_mean",{.002,pg::BumpScale::absolute}},
        {"model.jump_log_volatility",{.005}}, {"product.strike",{.005}}}};
    auto prepare=[&](const pg::PriceGradientConfiguration& selection) {
        return merton::prepare_merton_european_option_price_gradients(
            models,products,PriceConstruction::Aligned,{},selection);
    };
    auto launcher=merton::launch_merton_european_option_price_gradients_cuda<Side>;
    const auto plan=prepare(full);
    double maximum_independent_error=0.,maximum_independent_budget_fraction=0.;
    DeviceArray<merton::ModelParameters> dm(models);
    DeviceArray<product::EuropeanOptionParameters> dp(products);
    for (unsigned threads : {128U,256U}) {
        pg::LaunchConfiguration launch{pg::PricingMethod::monte_carlo,0U,3U,paths,threads,3U,719U};
        const auto solo=execute(prepare({{spot}}),launch,launcher);
        DeviceArray<float> old(12);
        merton::launch_merton_european_option_price_delta_cuda<Side>(models.data(),dm.data,3U,
            products.data(),dp.data,3U,PriceConstruction::Aligned,3U,0U,3U,paths,
            1.f/252.f,threads,3U,719U,{.01f},old.data,old.data+3,old.data+6,old.data+9);
        const auto reference=old.read();
        for (unsigned row=0;row<3;++row) {
            same(solo.price[row],reference[row],"Merton legacy price");
            same(solo.price_error[row],reference[3+row],"Merton legacy price SE");
            same(solo.gradient[row],reference[6+row],"Merton legacy delta");
            same(solo.gradient_error[row],reference[9+row],"Merton legacy delta SE");
        }
        const auto all=execute(plan,launch,launcher);
        const auto independent=independent_gradients<Side>(plan,all);
        maximum_independent_error=std::max(maximum_independent_error,independent.first);
        maximum_independent_budget_fraction=std::max(maximum_independent_budget_fraction,independent.second);
        batching(full,all,launch,prepare,launcher);
        for (std::size_t i=0;i<full.sensitivities.size();++i) {
            const auto one=execute(prepare({{full.sensitivities[i]}}),launch,launcher);
            for (std::size_t row=0;row<3;++row) {
                same(all.gradient[row*full.sensitivities.size()+i],one.gradient[row],
                    "Merton selected coordinate");
                same(all.gradient_error[row*full.sensitivities.size()+i],one.gradient_error[row],
                    "Merton selected coordinate SE");
            }
        }
    }
    std::cout << "Merton " << option_side_name(Side)
              << ": seven independent represented-stencil gradients passed; max_abs_error="
              << maximum_independent_error << ", max_budget_fraction="
              << maximum_independent_budget_fraction
              << "; variable-consumption parity and batches passed\n";
}

int main(int argc,char** argv) {
    try {
        std::size_t paths=65537;
        if (argc==2 && std::string_view(argv[1])=="--sanitizer") paths=513;
        else if (argc==2 && std::string_view(argv[1])=="--factorization-benchmark") {
            factorization_benchmark();return 0;
        }
        else if (argc!=1) throw std::invalid_argument("Usage: test [--sanitizer]");
        int devices=0;
        if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
        check<OptionSide::call>(paths);
        check<OptionSide::put>(paths);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
