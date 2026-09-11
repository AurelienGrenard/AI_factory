// Public Black-Scholes price/delta parity, analytic deltas and bump convergence.
#include "model/equity/markovian/black_scholes/product/european_option.cuh"
#include "model/equity/markovian/black_scholes/product/european_option_price_delta.cuh"
#include "tests/price_delta/cuda_test_support.cuh"

#include <bit>
#include <cmath>
#include <iostream>

namespace {
using namespace ai_factory::workbench;
namespace bs = model::equity::black_scholes;
using price_delta_test::require;
using price_delta_test::DeviceArray;

template<OptionSide Side>
void check_side() {
    constexpr std::size_t count = 3U;
    const bs::ModelParameters models[count]{
        {.75f,.03f,.01f,.2f}, {1.2f,.03f,.01f,.2f}, {1.4f,.03f,.01f,.2f}
    };
    const product::EuropeanOptionParameters products[count]{{1.f,252U},{1.2f,126U},{1.1f,252U}};
    DeviceArray<bs::ModelParameters> device_models(count);
    DeviceArray<product::EuropeanOptionParameters> device_products(count);
    DeviceArray<float> outputs(3U * count);
    check_cuda(cudaMemcpy(device_models.data, models, sizeof(models), cudaMemcpyHostToDevice), "models");
    check_cuda(cudaMemcpy(device_products.data, products, sizeof(products), cudaMemcpyHostToDevice), "products");
    bs::launch_black_scholes_european_option_cuda<Side>(device_models.data, count,
        device_products.data, count, PriceConstruction::Aligned, count, 0U, count,
        1.f/252.f, 128U, 1U, outputs.data);
    for (float width : {.01f,.005f,.002f}) {
        bs::launch_black_scholes_european_option_price_delta_cuda<Side>(models, device_models.data,
            count, device_products.data, count, PriceConstruction::Aligned, count, 0U, count,
            1.f/252.f, 128U, 1U, {width}, outputs.data + count, outputs.data + 2U*count);
        float results[3U * count];
        check_cuda(cudaMemcpy(results, outputs.data, sizeof(results), cudaMemcpyDeviceToHost), "results");
        for (std::size_t row = 0; row < count; ++row) {
            require(std::isfinite(results[row]) && std::isfinite(results[2U*count+row]), "non-finite result");
            require(std::bit_cast<unsigned int>(results[row])
                == std::bit_cast<unsigned int>(results[count+row]), "closed-form price bits differ");
            const auto& m = models[row];
            const double years = products[row].maturity_days / 252.0;
            const double d1 = (std::log(static_cast<double>(m.spot) / products[row].strike)
                + (m.risk_free_rate - m.dividend_yield + .5*m.volatility*m.volatility)*years)
                / (m.volatility*std::sqrt(years));
            const double cdf = .5*std::erfc(-d1/std::sqrt(2.0));
            const double expected = std::exp(-m.dividend_yield*years)
                * (cdf - (Side == OptionSide::put ? 1.0 : 0.0));
            require(std::abs(results[2U*count+row]-expected) < 2.e-4, "analytic delta mismatch");
        }
    }
    std::cout << option_side_name(Side) << ": central price bitwise; three bumps vs analytic delta passed\n";
}
}  // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        check_side<OptionSide::call>();
        check_side<OptionSide::put>();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
