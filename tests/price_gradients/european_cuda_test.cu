// Public selected-gradient checks: historical parity, analytic Greeks, subsets and launch guards.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/product/european_option_price_delta.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_delta.cuh"
#include "cuda_test_support.cuh"
#include <bit>
#include <iostream>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace bs = model::equity::black_scholes;
namespace hs = model::equity::heston;
std::size_t bs_paths = 1U << 17U;
std::size_t heston_paths = 4097U;
using namespace price_gradient_test;

template<OptionSide Side> void black_scholes() {
    const std::vector<bs::ModelParameters> models{{.75f,.03f,.01f,.2f},{1.2f,0.f,0.f,.3f},{1.4f,.04f,.02f,.25f}};
    const std::vector<product::EuropeanOptionParameters> products{{1.f,252U},{1.2f,126U},{1.1f,63U}};
    const pg::Sensitivity spot{"model.spot", {.005f}};
    const auto prepare = [&](const pg::PriceGradientConfiguration& selected) {
        return bs::prepare_black_scholes_european_option_price_gradients(models, products, PriceConstruction::Aligned, {}, selected);
    };
    auto launcher = bs::launch_black_scholes_european_option_price_gradients_cuda<Side>;
    pg::LaunchConfiguration launch{pg::PricingMethod::closed_form, 0U, 3U, bs_paths, 128U, 1U, 719U};
    const auto solo = execute(prepare({{spot}}), launch, launcher, true);
    DeviceArray<bs::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    DeviceArray<float> legacy(6U);
    bs::launch_black_scholes_european_option_price_delta_cuda<Side>(models.data(), device_models.data, 3U,
        device_products.data, 3U, PriceConstruction::Aligned, 3U, 0U, 3U, 1.f/252.f, 128U, 1U, {.01f}, legacy.data, legacy.data+3U);
    const auto reference = legacy.read();
    for (unsigned int row = 0; row < 3; ++row) {
        same(solo.price[row], reference[row], "Black-Scholes central price parity");
        same(solo.gradient[row], reference[3U+row], "Black-Scholes spot delta parity");
    }
    const pg::PriceGradientConfiguration full{{spot,
        {"model.volatility", {.002f}}, {"product.strike", {.002f}},
        {"model.risk_free_rate", {.0001f, pg::BumpScale::absolute}},
        {"model.dividend_yield", {.0001f, pg::BumpScale::absolute}},
        {"product.maturity_years", {1.f/504.f, pg::BumpScale::absolute}}}};
    const auto analytical = execute(prepare(full), launch, launcher);
    const double sign = Side == OptionSide::call ? 1.0 : -1.0;
    for (unsigned int row = 0; row < 3; ++row) {
        same(analytical.price[row], solo.price[row], "Adding gradients changed closed-form price");
        same(analytical.gradient[row*6U], solo.gradient[row], "Adding gradients changed closed-form delta");
        const auto& m = models[row];
        const double t = products[row].maturity_days/252.0, strike = products[row].strike;
        const double d1 = (std::log(m.spot/strike)+(m.risk_free_rate-m.dividend_yield+.5*m.volatility*m.volatility)*t)/(m.volatility*std::sqrt(t));
        const double d2 = d1-m.volatility*std::sqrt(t);
        const auto cdf = [](double x) { return .5*std::erfc(-x/std::sqrt(2.0)); };
        const double dq = std::exp(-m.dividend_yield*t), dr = std::exp(-m.risk_free_rate*t);
        const double density = std::exp(-.5*d1*d1)/std::sqrt(2.0*std::acos(-1.0));
        const double expected[6]{sign*dq*cdf(sign*d1), m.spot*dq*density*std::sqrt(t),
            -sign*dr*cdf(sign*d2), sign*strike*t*dr*cdf(sign*d2), -sign*m.spot*t*dq*cdf(sign*d1),
            m.spot*dq*density*m.volatility/(2*std::sqrt(t))
                + sign*m.risk_free_rate*strike*dr*cdf(sign*d2)-sign*m.dividend_yield*m.spot*dq*cdf(sign*d1)};
        for (unsigned int i = 0; i < 6; ++i)
            require(std::abs(analytical.gradient[row*6U+i]-expected[i]) < 1.5e-3, "Analytical gradient mismatch.");
    }
    launch.method = pg::PricingMethod::monte_carlo;
    const auto mc = execute(prepare(full), launch, launcher, true);
    batching(full,mc,launch,prepare,launcher);
    const auto mc_solo = execute(prepare({{spot}}), launch, launcher);
    for (unsigned int row = 0; row < 3; ++row) {
        same(mc.price[row], mc_solo.price[row], "Adding gradients changed MC price");
        same(mc.price_error[row], mc_solo.price_error[row], "Adding gradients changed MC price error");
        same(mc.gradient[row*6U], mc_solo.gradient[row], "Adding gradients changed MC delta");
        same(mc.gradient_error[row*6U], mc_solo.gradient_error[row], "Adding gradients changed MC delta error");
        for (unsigned int i = 0; i < 6; ++i)
            require(std::abs(mc.gradient[row*6U+i]-analytical.gradient[row*6U+i])
                < 6*mc.gradient_error[row*6U+i]+.002, "MC gradient differs from analytical value beyond paired noise budget.");
    }
    for (unsigned int i = 1; i < 6; ++i) {
        const auto single = execute(prepare({{full.sensitivities[i]}}),launch,launcher);
        for (unsigned int row = 0; row < 3; ++row) {
            same(mc.gradient[row*6U+i],single.gradient[row],full.sensitivities[i].parameter.c_str());
            same(mc.gradient_error[row*6U+i],single.gradient_error[row],full.sensitivities[i].parameter.c_str());
        }
    }
    const auto reversed = execute(prepare({{{"product.strike", {.002f}}, spot}}), launch, launcher);
    for (unsigned int row = 0; row < 3; ++row) {
        same(reversed.gradient[row*2U+1U], mc_solo.gradient[row], "Selection order changed spot delta");
        same(reversed.gradient[row*2U], mc.gradient[row*6U+2U], "Selection order changed strike gradient");
    }
    const auto empty = execute(prepare({}), launch, launcher);
    for (unsigned int row = 0; row < 3; ++row) same(empty.price[row], mc_solo.price[row], "Empty selection changed price");
    std::cout << "Black-Scholes " << option_side_name(Side) << ": CF parity, six analytical/MC gradients and selection invariance passed\n";
}

template<OptionSide Side> void heston() {
    const std::vector<hs::ModelParameters> models{{.75f,.03f,.01f,.04f,1.5f,.04f,.3f,-.7f},{1.2f,0.f,0.f,.06f,.8f,.04f,.4f,-.3f},{1.4f,.01f,0.f,0.f,1.f,.04f,.3f,1.f}};
    const std::vector<product::EuropeanOptionParameters> products{{.8f,16U},{1.2f,12U},{1.1f,8U}};
    auto prepare = [&](const pg::PriceGradientConfiguration& selection) {
        return hs::prepare_heston_european_option_price_gradients(models, products, PriceConstruction::Aligned, {}, selection);
    };
    const pg::Sensitivity spot{"model.spot", {.005f}};
    auto launcher = hs::launch_heston_european_option_price_gradients_cuda<Side>;
    DeviceArray<hs::ModelParameters> device_models(models);
    DeviceArray<product::EuropeanOptionParameters> device_products(products);
    for (unsigned int threads : {64U, 128U, 256U}) {
        pg::LaunchConfiguration launch{pg::PricingMethod::monte_carlo,0U,3U,heston_paths,threads,1U,713U};
        const auto solo = execute(prepare({{spot}}), launch, launcher, true);
        DeviceArray<float> old(12U);
        hs::launch_heston_european_option_price_delta_cuda<Side>(models.data(),device_models.data,3U,
            products.data(),device_products.data,3U,PriceConstruction::Aligned,3U,0U,3U,heston_paths,1.f/504.f,2U,
            threads,1U,713U,{.01f},old.data,old.data+3U,old.data+6U,old.data+9U);
        const auto reference = old.read();
        for (unsigned int row = 0; row < 3; ++row) {
            same(solo.price[row],reference[row],"Heston price parity");
            same(solo.price_error[row],reference[3U+row],"Heston price error parity");
            same(solo.gradient[row],reference[6U+row],"Heston delta parity");
            same(solo.gradient_error[row],reference[9U+row],"Heston delta error parity");
        }
        const auto extended = execute(prepare({{{"model.initial_variance", {.001f,pg::BumpScale::absolute}},
            {"model.rho", {.002f,pg::BumpScale::absolute}},spot,{"product.strike",{.002f}}}}),launch,launcher);
        for (unsigned int row = 0; row < 3; ++row) {
            same(extended.price[row],solo.price[row],"Heston subset price parity");
            same(extended.gradient[row*4U+2U],solo.gradient[row],"Heston subset delta parity");
            same(extended.gradient_error[row*4U+2U],solo.gradient_error[row],"Heston subset delta error parity");
        }
        const pg::PriceGradientConfiguration full{{spot,
            {"model.risk_free_rate", {.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield", {.0001,pg::BumpScale::absolute}},
            {"model.initial_variance", {.001,pg::BumpScale::absolute}},
            {"model.kappa", {.005}}, {"model.theta", {.005}}, {"model.gamma", {.005}},
            {"model.rho", {.002,pg::BumpScale::absolute}}, {"product.strike", {.002}},
            {"product.maturity_years", {1.f/504.f,pg::BumpScale::absolute}}}};
        const auto all = execute(prepare(full),launch,launcher);
        const auto empty = execute(prepare({}),launch,launcher);
        for (unsigned int row = 0; row < 3; ++row) {
            same(all.price[row],solo.price[row],"Heston full-selection price parity");
            same(empty.price[row],solo.price[row],"Heston empty-selection price parity");
            same(all.gradient[row*10U],solo.gradient[row],"Heston full-selection delta parity");
            same(all.gradient_error[row*10U],solo.gradient_error[row],"Heston full-selection delta error parity");
        }
        batching(full,all,launch,prepare,launcher);
        if (threads == 128U) {
            for (unsigned int i = 1; i < 10; ++i) {
                const auto single = execute(prepare({{full.sensitivities[i]}}),launch,launcher);
                for (unsigned int row = 0; row < 3; ++row) {
                    same(all.gradient[row*10U+i],single.gradient[row],full.sensitivities[i].parameter.c_str());
                    same(all.gradient_error[row*10U+i],single.gradient_error[row],full.sensitivities[i].parameter.c_str());
                }
            }
        }
    }
    std::cout << "Heston " << option_side_name(Side) << ": historical four-output parity, ten coordinates, reordered subsets, boundaries and grid maturities passed\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--sanitizer") {
            // Keep every model/side/batch/tail/geometry case and multiple path
            // iterations, including a partial final iteration at 256 threads.
            bs_paths = 1025U;
            heston_paths = 513U;
            std::cout << "Sanitizer fixture: BS paths=1025, Heston paths=513; complete case matrix\n";
        } else if (argc != 1) throw std::invalid_argument("Usage: test_price_gradients_european_cuda [--sanitizer]");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        black_scholes<OptionSide::call>(); black_scholes<OptionSide::put>();
        heston<OptionSide::call>(); heston<OptionSide::put>();
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
