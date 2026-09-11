// CIR++ curve fitting, conditional analytics, exact-factor samples and LSM integration.
// JSON-lines output is consumed by the explicit independent-reference diagnostic.
#include "model/fixed_income/cir_plus_plus/nelson_siegel/analytics_impl.cuh"
#include "model/fixed_income/cir_plus_plus/svensson/analytics_impl.cuh"
#include "model/fixed_income/cir_plus_plus/product/nelson_siegel/bermudan_swaption.cuh"
#include "model/fixed_income/cir_plus_plus/product/svensson/bermudan_swaption.cuh"
#include "model/fixed_income/cir_plus_plus/sample.cuh"
#include "model/fixed_income/cir/sample.cuh"
#include "product/european_swaption/schedule.cuh"
#include "common/fixed_income/jamshidian_cooperative.cuh"
#include "tools/cuda/pricing_runner.cuh"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
#include <nlohmann/json.hpp>

namespace wb = ai_factory::workbench;
namespace cir = wb::model::fixed_income::cir;
namespace cirpp = wb::model::fixed_income::cir_plus_plus;
namespace fitted = cirpp::fitted;
namespace cuda = wb::offline::cuda;
using Json = nlohmann::ordered_json;
using CurveNS = wb::curve::nelson_siegel::NelsonSiegelParameters;
using CurveSV = wb::curve::svensson::SvenssonParameters;
using wb::SwaptionSide;

const std::vector<cirpp::ModelParameters> models{
    {{.5f,.04f,.1f},.03f}, {{.1f,.02f,.3f},.001f},
    {{2.f,.12f,.04f},.2f}, {{.03f,.001f,.05f},0.f},
    {{.8f,.08f,.35f},.08f}, {{.005f,.0001f,.001f},0.f},
    {{2.5f,.2f,.8f},.2f}, {{.03f,.04f,.15f},.002f}
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void close(float actual, float expected, float allowance, const char* message) {
    require(std::isfinite(actual) && std::isfinite(expected)
        && std::abs(actual - expected) <= allowance, message);
}

// An endogenous CIR curve must collapse CIR++ back to exactly the base model.
struct CirCurve {
    using Parameters = cir::ModelParameters;
    __device__ static float log_discount_factor(const Parameters& model, float maturity_years) {
        return cir::log_zero_coupon_bond(model, model.initial_state, 0.0f, maturity_years);
    }
};

struct AnalyticsOutput {
    float fitted_bond, market_bond, initial_rate, initial_forward;
    float conditional_bond, call, put, parity;
    float payer, receiver, swap, base_call, unshifted_call;
    float shift, shift_derivative;
};

template<typename Provider>
__global__ void analytics_kernel(
    const cirpp::ModelParameters* factors, typename Provider::Parameters curve,
    std::size_t count, AnalyticsOutput* output
) {
    const std::size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const auto factor = factors[row];
    const auto model = fitted::compose_fitted_model<Provider>(factor, curve);
    const auto unshifted = fitted::compose_fitted_model<CirCurve>(factor, factor);
    constexpr float time_years = 2.0f, expiry_years = 5.0f, maturity_years = 9.0f;
    constexpr float state = .08f, strike = .8f;
    const wb::product::RegularEuropeanSwaptionScheduleView schedule{2.5f,.5f,.5f,8U};
    auto& result = output[row];
    result.fitted_bond = fitted::zero_coupon_bond(model, factor.initial_state, 0.0f, 50.0f);
    result.market_bond = expf(Provider::log_discount_factor(curve, 50.0f));
    result.initial_rate = fitted::short_rate(model, factor.initial_state, 0.0f);
    result.initial_forward = Provider::instantaneous_forward(curve, 0.0f);
    result.conditional_bond = fitted::zero_coupon_bond(model, state, time_years, maturity_years);
    result.call = fitted::zero_coupon_bond_call_price(model, state, time_years,
        expiry_years, maturity_years, strike);
    result.put = fitted::zero_coupon_bond_put_price(model, state, time_years,
        expiry_years, maturity_years, strike);
    result.parity = result.conditional_bond - strike
        * fitted::zero_coupon_bond(model, state, time_years, expiry_years);
    result.payer = fitted::european_payer_swaption_price(model, factor.initial_state,
        0.0f, 2.0f, .035f, schedule);
    result.receiver = fitted::european_receiver_swaption_price(model, factor.initial_state,
        0.0f, 2.0f, .035f, schedule);
    result.swap = fitted::payer_swap_value(model, factor.initial_state, 0.0f, 2.0f, .035f, schedule);
    result.base_call = cir::zero_coupon_bond_call_price(factor, state, time_years,
        expiry_years, maturity_years, strike);
    result.unshifted_call = fitted::zero_coupon_bond_call_price(unshifted, state, time_years,
        expiry_years, maturity_years, strike);
    result.shift = fitted::short_rate_shift(model, time_years);
    constexpr float h = .01f;
    result.shift_derivative = (fitted::shift_integral(model, 0.f, time_years+h)
        - fitted::shift_integral(model, 0.f, time_years-h)) / (2*h);
}

template<typename Provider>
void check_analytics(typename Provider::Parameters curve, const char* curve_name, Json curve_json) {
    cuda::DeviceBuffer<cirpp::ModelParameters> factors(models.size());
    cuda::DeviceBuffer<AnalyticsOutput> output(models.size());
    factors.copy_from(models.data());
    analytics_kernel<Provider><<<1,128>>>(factors.data(), curve, models.size(), output.data());
    wb::check_cuda(cudaGetLastError(), "CIR++ analytics test kernel");
    std::vector<AnalyticsOutput> results(models.size());
    output.copy_to(results.data());
    for (std::size_t row=0; row<results.size(); ++row) {
        const auto& r = results[row];
        close(r.fitted_bond,r.market_bond,2e-6f*std::max(1.f,r.market_bond),"CIR++ curve fit failed");
        close(r.initial_rate,r.initial_forward,4e-8f,"CIR++ initial rate is not the curve forward");
        close(r.call-r.put,r.parity,2e-6f,"CIR++ bond-option parity failed");
        close(r.payer-r.receiver,r.swap,3e-6f,"CIR++ swaption parity failed");
        close(r.base_call,r.unshifted_call,2e-7f,"Zero-shift CIR++ differs from CIR");
        close(r.shift,r.shift_derivative,1e-5f,"CIR++ shift is not the derivative of its integral");
        require(r.call>=0 && r.put>=0 && r.payer>=0 && r.receiver>=0,"Negative CIR++ option");
        const auto& m = models[row];
        std::cout << Json{{"kind","analytics"},{"curve",curve_name},{"curve_parameters",curve_json},
            {"model",{{"mean_reversion",m.process.mean_reversion},{"long_term_mean",m.process.long_term_mean},
                      {"volatility",m.process.volatility},{"initial_state",m.initial_state}}},
            {"conditional_bond",r.conditional_bond},{"call",r.call},{"put",r.put},
            {"payer",r.payer},{"receiver",r.receiver}}.dump() << '\n';
    }
}

// Catalogue stress row 909: zero coupon, 600 payments, one-day exercise.
__global__ void zero_strike_regression_kernel(float* result) {
    using Provider = wb::curve::nelson_siegel::AnalyticsProvider;
    const cirpp::ModelParameters factor{{2.439925909f,.074310377f,.785255373f},.008517840f};
    const CurveNS curve{.048722975f,.008850992f,-.132413f,2.054715157f};
    const auto model = fitted::compose_fitted_model<Provider>(factor,curve);
    constexpr float exercise = 1.0f / 252.0f;
    const wb::product::RegularEuropeanSwaptionScheduleView schedule{
        exercise+21.0f/252.0f,21.0f/252.0f,1.0f/12.0f,600U};
    const float maturity = schedule.payment_time(599U);
    const fitted::AnalyticsProvider<Provider> provider{};
    if (threadIdx.x == 0U) {
        result[0] = wb::fixed_income::jamshidian_state_boundary(provider,model,exercise,0.f,schedule);
        result[1] = fitted::european_payer_swaption_price(model,factor.initial_state,0.f,exercise,0.f,schedule);
        result[2] = fitted::zero_coupon_bond_put_price(model,factor.initial_state,0.f,exercise,maturity,1.f);
        result[3] = fitted::european_receiver_swaption_price(model,factor.initial_state,0.f,exercise,0.f,schedule);
        result[4] = fitted::zero_coupon_bond_call_price(model,factor.initial_state,0.f,exercise,maturity,1.f);
        result[5] = wb::fixed_income::evaluate_jamshidian_boundary(provider,model,exercise,0.f,schedule,result[0]).residual;
    }
    __shared__ float storage[3U*600U];
    const float payer = wb::fixed_income::cooperative_european_swaption_price<SwaptionSide::payer>(
        provider,model,factor.initial_state,0.f,exercise,0.f,schedule,
        reinterpret_cast<std::byte*>(storage),600U);
    if (threadIdx.x == 0U) result[6] = payer;
    __syncthreads();
    const float receiver = wb::fixed_income::cooperative_european_swaption_price<SwaptionSide::receiver>(
        provider,model,factor.initial_state,0.f,exercise,0.f,schedule,
        reinterpret_cast<std::byte*>(storage),600U);
    if (threadIdx.x == 0U) result[7] = receiver;
}

void check_zero_strike_regression() {
    cuda::DeviceBuffer<float> output(8U);
    zero_strike_regression_kernel<<<1,128>>>(output.data());
    wb::check_cuda(cudaGetLastError(),"CIR++ zero-strike regression kernel");
    std::array<float,8> result{};
    output.copy_to(result.data());
    require(std::isfinite(result[0]),"CIR++ rounded Jamshidian bracket rejected its valid endpoint");
    close(result[5],0.f,wb::fixed_income::kJamshidianResidualTolerance,"CIR++ root residual was relaxed");
    close(result[1],result[2],2e-6f,"CIR++ zero-strike swaption is not a unit-strike bond put");
    close(result[3],result[4],2e-6f,"CIR++ zero-strike swaption is not a unit-strike bond call");
    close(result[6],result[2],2e-6f,"Cooperative CIR++ zero-strike payer regression");
    close(result[7],result[4],2e-6f,"Cooperative CIR++ zero-strike receiver regression");
}

void check_samples() {
    constexpr std::size_t paths=1024, count=8*paths;
    cuda::DeviceBuffer<cirpp::ModelParameters> dm(models.size());
    cuda::DeviceBuffer<float> base(count), shifted(count);
    dm.copy_from(models.data());
    cir::launch_cir_terminal_samples_cuda(dm.data(),models.size(),paths,252,0,count,128,8,8821,base.data());
    cirpp::launch_cir_plus_plus_terminal_samples_cuda(dm.data(),models.size(),paths,252,0,count,128,8,8821,shifted.data());
    std::vector<float> a(count), b(count);
    base.copy_to(a.data()); shifted.copy_to(b.data());
    require(a==b,"CIR++ sampler changed the underlying CIR trajectory mapping");
    for (auto x:b) require(std::isfinite(x) && x>=0,"Invalid CIR++ factor sample");
    // Both caller batching and geometry must preserve path-indexed Philox counters.
    for (std::size_t offset=0; offset<count; offset+=paths)
        cirpp::launch_cir_plus_plus_terminal_samples_cuda(dm.data(),models.size(),paths,252,
            offset,paths,256,1,8821,shifted.data());
    shifted.copy_to(b.data());
    require(a==b,"CIR++ sample batching changed trajectories");
}

template<typename Curve, typename Launcher>
void check_bermudan(Curve curve, const char* curve_name, Launcher launch, std::size_t paths) {
    const std::vector<cirpp::ModelParameters> factors{models[0],models[1],models[6]};
    const std::vector<Curve> curves(factors.size(),curve);
    const std::vector<wb::product::BermudanSwaptionParameters> products{
        {1,.035f,.5f,126,126,6,4}, {1,.035f,.5f,126,126,6,4}, {1,.035f,.5f,126,126,6,4}
    };
    cuda::DeviceBuffer<cirpp::ModelParameters> dm(factors.size());
    cuda::DeviceBuffer<Curve> dc(curves.size());
    cuda::DeviceBuffer<wb::product::BermudanSwaptionParameters> dp(products.size());
    cuda::DeviceBuffer<float> price(products.size()), errors(products.size());
    dm.copy_from(factors.data()); dc.copy_from(curves.data()); dp.copy_from(products.data());
    for (auto side : {SwaptionSide::payer,SwaptionSide::receiver}) {
        std::vector<float> first(products.size()), se(products.size()), replay(products.size());
        launch(side,dm.data(),dc.data(),products.data(),dp.data(),products.size(),paths,price.data(),errors.data());
        price.copy_to(first.data()); errors.copy_to(se.data());
        launch(side,dm.data(),dc.data(),products.data(),dp.data(),products.size(),paths,price.data(),errors.data());
        price.copy_to(replay.data());
        require(first==replay,"CIR++ LSM replay changed prices");
        for (std::size_t row=0;row<first.size();++row) {
            require(std::isfinite(first[row]) && first[row]>=0 && std::isfinite(se[row]) && se[row]>=0,
                "Invalid CIR++ LSM price or standard error");
            const auto& m=factors[row];
            std::cout << Json{{"kind","bermudan"},{"curve",curve_name},{"row",row},
                {"side",side==SwaptionSide::payer?"payer":"receiver"},{"price",first[row]},
                {"standard_error",se[row]},{"paths",paths},
                {"model",{{"mean_reversion",m.process.mean_reversion},{"long_term_mean",m.process.long_term_mean},
                          {"volatility",m.process.volatility},{"initial_state",m.initial_state}}}}.dump() << '\n';
        }
    }
}

int main(int argc,char** argv) try {
    std::size_t paths=1U<<20U;
    if(argc==2 && std::string_view(argv[1])=="--smoke-test") paths=4096;
    else if(argc!=1) throw std::invalid_argument("Expected no arguments or --smoke-test");
    int count=0;
    const auto status=cudaGetDeviceCount(&count);
    if(status==cudaErrorNoDevice || status==cudaErrorInsufficientDriver || count==0) return 77;
    wb::check_cuda(status,"CIR++ test device discovery");
    check_zero_strike_regression();
    for (float level : {.03f,-.01f,.12f}) {
        check_analytics<wb::curve::nelson_siegel::AnalyticsProvider>(
            CurveNS{level,-.01f,.02f,2.f},"nelson_siegel",
            {{"beta0",level},{"beta1",-.01f},{"beta2",.02f},{"tau",2.f}});
        check_analytics<wb::curve::svensson::AnalyticsProvider>(
            CurveSV{level,-.01f,.02f,.01f,2.f,5.f},"svensson",
            {{"beta0",level},{"beta1",-.01f},{"beta2",.02f},{"beta3",.01f},{"tau1",2.f},{"tau2",5.f}});
    }
    check_samples();
    check_bermudan(CurveNS{.03f,-.01f,.02f,2.f},"nelson_siegel",
        [](auto side,auto m,auto c,auto hp,auto p,auto n,auto paths,auto out,auto se) {
            const auto run=[&]<SwaptionSide Side>() {
                cirpp::nelson_siegel::launch_cir_plus_plus_nelson_siegel_bermudan_swaption_cuda<Side>(
                    m,n,c,n,hp,p,n,wb::PriceConstruction::Aligned,n,paths,1.f/252,128,16,90217,out,se);
            };
            if(side==SwaptionSide::payer) run.template operator()<SwaptionSide::payer>();
            else run.template operator()<SwaptionSide::receiver>();
        },paths);
    check_bermudan(CurveSV{.03f,-.01f,.02f,.01f,2.f,5.f},"svensson",
        [](auto side,auto m,auto c,auto hp,auto p,auto n,auto paths,auto out,auto se) {
            const auto run=[&]<SwaptionSide Side>() {
                cirpp::svensson::launch_cir_plus_plus_svensson_bermudan_swaption_cuda<Side>(
                    m,n,c,n,hp,p,n,wb::PriceConstruction::Aligned,n,paths,1.f/252,128,16,90217,out,se);
            };
            if(side==SwaptionSide::payer) run.template operator()<SwaptionSide::payer>();
            else run.template operator()<SwaptionSide::receiver>();
        },paths);
} catch (const std::exception& error) {
    std::cerr << "CIR++ test failed: " << error.what() << '\n';
    return 1;
}
