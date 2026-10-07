// Contract checks for the OU grid-monitored zero-coupon bond barrier call.
#include "model/fixed_income/ornstein_uhlenbeck/product/zero_coupon_bond_up_and_out.cuh"
#include "tools/cuda/pricing_runner.cuh"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace wb = ai_factory::workbench;
namespace ou = wb::model::fixed_income::ornstein_uhlenbeck;
using Product = wb::product::ZeroCouponBondUpAndOutParameters;
using Model = ou::ModelParameters;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Result { float price; float se; };
Result price(const Model& model, const Product& product,
             std::uint32_t steps_per_day, std::size_t paths,
             std::uint64_t seed = 20261006ULL) {
    wb::offline::cuda::DeviceBuffer<Model> dm(1U);
    wb::offline::cuda::DeviceBuffer<Product> dp(1U);
    wb::offline::cuda::DeviceBuffer<float> dv(1U), de(1U);
    dm.copy_from(&model); dp.copy_from(&product);
    ou::launch_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_cuda(
        dm.data(), 1U, &product, dp.data(), 1U, wb::PriceConstruction::Aligned,
        1U, 0U, 1U, paths, steps_per_day, 256U, 1U, seed, dv.data(), de.data());
    Result result{};
    dv.copy_to(&result.price); de.copy_to(&result.se);
    return result;
}

double loading(double a, double t) { return -std::expm1(-a*t)/a; }
double integrated_variance(double a, double sigma, double t) {
    return sigma*sigma/(a*a)*(t-2*loading(a,t)+loading(2*a,t));
}
double p0(double a, double sigma, double x0, double maturity) {
    return std::exp(-loading(a,maturity)*x0
                    +0.5*integrated_variance(a,sigma,maturity));
}
double phi(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
double independent_bond_call(const Model& model, const Product& p) {
    const double a=model.process.mean_reversion, sigma=model.process.volatility;
    const double t=p.option_expiry_days/252.0, u=p.bond_maturity_days/252.0;
    const double p0u=p0(a,sigma,model.initial_state,u);
    const double p0t=p0(a,sigma,model.initial_state,t);
    const double variance=sigma*sigma*(1-std::exp(-2*a*t))/(2*a);
    const double volatility=loading(a,u-t)*std::sqrt(variance);
    if (volatility==0.0) return p.notional*std::max(p0u-p.strike*p0t,0.0);
    const double d1=(std::log(p0u/(p.strike*p0t))+0.5*volatility*volatility)/volatility;
    return p.notional*(p0u*phi(d1)-p.strike*p0t*phi(d1-volatility));
}

int main() {
    const Model model{{0.5f,0.02f},0.03f};
    Product p{1.0f,0.90f,10.0f,252U,504U};
    const double initial_bond=p0(0.5,0.02,0.03,2.0);
    p.barrier=static_cast<float>(initial_bond*0.99);
    const auto knocked=price(model,p,1U,16384U);
    require(knocked.price==0.0f && knocked.se==0.0f,
            "barrier breached at t=0 must return exact zero");

    p.barrier=10.0f;
    const auto european=price(model,p,1U,1U<<20U);
    const double reference=independent_bond_call(model,p);
    require(std::fabs(european.price-reference)<5*european.se+2e-5,
            "remote barrier must converge to independent OU bond call");

    const Model deterministic{{0.5f,0.0f},0.03f};
    Product deterministic_product{2.5f,0.90f,10.0f,126U,252U};
    const auto deterministic_result=price(deterministic,deterministic_product,2U,4096U);
    const double expected=independent_bond_call(deterministic,deterministic_product);
    require(std::fabs(deterministic_result.price-expected)<2e-5,
            "deterministic T/U calendar, notional and integrated discount disagree");

    Product invalid=p; invalid.bond_maturity_days=invalid.option_expiry_days;
    bool rejected=false;
    try { (void)price(model,invalid,1U,4096U); }
    catch (const std::invalid_argument&) { rejected=true; }
    require(rejected,"T >= U must be rejected");

    p.strike=0.90f;
    p.barrier=static_cast<float>(initial_bond*1.01);
    const auto daily=price(model,p,1U,1U<<19U);
    const auto half_daily=price(model,p,2U,1U<<19U);
    const auto quarter_daily=price(model,p,4U,1U<<19U);
    require(daily.price+4*std::hypot(daily.se,half_daily.se)>=half_daily.price
            && half_daily.price+4*std::hypot(half_daily.se,quarter_daily.se)>=quarter_daily.price,
            "grid refinement should not raise up-and-out value beyond MC error");
    std::cout << "initial_bond=" << initial_bond
              << " european=" << european.price << " se=" << european.se
              << " closed_form=" << reference
              << " deterministic=" << deterministic_result.price
              << " deterministic_reference=" << expected
              << " grid_1=" << daily.price << " se_1=" << daily.se
              << " grid_2=" << half_daily.price << " se_2=" << half_daily.se
              << " grid_4=" << quarter_daily.price << " se_4=" << quarter_daily.se
              << '\n';
}
