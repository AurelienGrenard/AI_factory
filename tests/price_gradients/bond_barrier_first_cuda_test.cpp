// First-order CRN checks for a grid-monitored zero-coupon bond barrier call.
#include "model/fixed_income/ornstein_uhlenbeck/product/zero_coupon_bond_up_and_out_price_gradients.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/product/zero_coupon_bond_up_and_out.cuh"
#include "tests/price_gradients/cuda_test_support.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace ou = model::fixed_income::ornstein_uhlenbeck;
namespace pg = price_gradients;
using Model = ou::ModelParameters;
using Product = product::ZeroCouponBondUpAndOutParameters;

constexpr std::size_t kPaths = 1U << 14U;
constexpr std::uint64_t kSeed = 20261013ULL;

price_gradient_test::Results evaluate(const Model& model, const Product& product) {
    const pg::PriceGradientConfiguration selection{{
        {"product.strike", {0.005f}},
        {"product.barrier", {0.005f}},
        {"product.maturity_years", {1.0f / 504.0f, pg::BumpScale::absolute}},
    }};
    const auto plan = ou::prepare_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_price_gradients(
        std::vector<Model>{model}, std::vector<Product>{product},
        PriceConstruction::Aligned, {1.0f / 504.0f, 2U}, selection
    );
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, 1U, kPaths, 128U,
        selection.sensitivities.size(), kSeed, 1U
    };
    return price_gradient_test::execute(
        plan, launch,
        [](const auto& host, auto inputs, auto stencils,
           const auto& configuration, auto outputs) {
            ou::launch_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_price_gradients_cuda(
                host, inputs, stencils, configuration, outputs
            );
        }
    );
}

float price_only(const Model& model, const Product& product) {
    price_gradient_test::DeviceArray<Model> device_models(std::vector<Model>{model});
    price_gradient_test::DeviceArray<Product> device_products(
        std::vector<Product>{product}
    );
    price_gradient_test::DeviceArray<float> prices(1U), errors(1U);
    ou::launch_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_cuda(
        device_models.data, 1U, &product, device_products.data, 1U,
        PriceConstruction::Aligned, 1U, 0U, 1U, kPaths,
        2U, 128U, 1U, kSeed, prices.data, errors.data
    );
    return prices.read()[0U];
}

double loading(double a, double t) { return -std::expm1(-a*t)/a; }
double integrated_variance(double a, double sigma, double t) {
    return sigma*sigma/(a*a)*(t-2*loading(a,t)+loading(2*a,t));
}
double p0(double a, double sigma, double x0, double t) {
    return std::exp(-loading(a,t)*x0+0.5*integrated_variance(a,sigma,t));
}
double phi(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }
double independent_call(const Model& model, double notional, double strike,
                        double expiry, double bond_maturity) {
    const double a=model.process.mean_reversion;
    const double sigma=model.process.volatility;
    const double p0u=p0(a,sigma,model.initial_state,bond_maturity);
    const double p0t=p0(a,sigma,model.initial_state,expiry);
    const double variance=sigma*sigma*(1-std::exp(-2*a*expiry))/(2*a);
    const double vol=loading(a,bond_maturity-expiry)*std::sqrt(variance);
    const double d1=(std::log(p0u/(strike*p0t))+0.5*vol*vol)/vol;
    return notional*(p0u*phi(d1)-strike*p0t*phi(d1-vol));
}

void near_mc(double actual, double expected, double standard_error,
             double floor, const char* label) {
    if (std::fabs(actual-expected) > 6.0*standard_error+floor) {
        std::cerr << label << ": actual=" << actual << " reference="
                  << expected << " se=" << standard_error << '\n';
        throw std::runtime_error(label);
    }
}
}  // namespace

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices==0) return 77;
    const Model model{{0.5f,0.02f},0.03f};
    Product product{1.0f,0.90f,10.0f,126U,252U};
    const double expiry=product.option_expiry_days/252.0;
    const double bond_maturity=product.bond_maturity_days/252.0;
    const auto unknocked=evaluate(model,product);
    const double reference=independent_call(
        model, product.notional, product.strike, expiry, bond_maturity
    );
    near_mc(unknocked.price[0],reference,unknocked.price_error[0],2e-4,
            "Remote barrier price disagrees with independent OU bond call");
    const double legacy_price=price_only(model,product);
    price_gradient_test::require(
        std::fabs(unknocked.price[0]-legacy_price)
            <= 2e-6+2e-5*std::max(
                std::fabs(static_cast<double>(unknocked.price[0])),
                std::fabs(legacy_price)
            ),
        "Barrier gradient central price disagrees with price-only replay."
    );

    const double p0u=p0(0.5,0.02,0.03,bond_maturity);
    const double p0t=p0(0.5,0.02,0.03,expiry);
    const double vol=loading(0.5,bond_maturity-expiry)*
        std::sqrt(0.02*0.02*(1-std::exp(-expiry))/(2*0.5));
    const double d2=(std::log(p0u/(product.strike*p0t))-
        0.5*vol*vol)/vol;
    near_mc(unknocked.gradient[0],-product.notional*p0t*phi(d2),
            unknocked.gradient_error[0],2e-3,"Strike derivative");
    price_gradient_test::require(
        unknocked.gradient[1] == 0.0f,
        "A remote barrier has zero first derivative in the unknocked region."
    );
    const double half_day=1.0/504.0;
    const double u_reference=(
        independent_call(model,product.notional,product.strike,expiry,
                         bond_maturity+half_day)
        - independent_call(model,product.notional,product.strike,expiry,
                           bond_maturity-half_day)
    )/(2*half_day);
    near_mc(unknocked.gradient[2],u_reference,
            unknocked.gradient_error[2],3e-3,"Bond maturity derivative");

    product.barrier=0.5f;
    const auto knocked=evaluate(model,product);
    price_gradient_test::require(
        knocked.price[0] == 0.0f && knocked.price_error[0] == 0.0f,
        "A barrier breached at time zero must have zero price."
    );
    for (const float gradient: knocked.gradient) {
        price_gradient_test::require(
            gradient == 0.0f,
            "A barrier breached at time zero must have zero first derivatives."
        );
    }
    std::cout << "OU barrier first-order CRN and independent bond-call checks passed\n";
}
