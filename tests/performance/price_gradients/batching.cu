// Fixed-workload public-API comparison; keep allocations outside timed launches.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "tests/performance/benchmark_support.cuh"
#include <bit>
#include "benchmark_support.cuh"

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace bs = model::equity::black_scholes;
namespace hs = model::equity::heston;
using namespace gradient_benchmark;

int main(int argc, char** argv) {
    try {
        const unsigned width = argc > 1 ? std::stoul(argv[1]) : pg::kDefaultSensitivityBatchSize;
        const unsigned threads = argc > 2 ? std::stoul(argv[2]) : pg::kDefaultThreadsPerBlock;
        const std::size_t rows = argc > 3 ? std::stoul(argv[3]) : 64U;
        if (argc > 4 || rows == 0U) throw std::invalid_argument("Usage: benchmark [B [THREADS [ROWS]]]");
        const std::vector<product::EuropeanOptionParameters> products(rows,{1.f,126U});
        const pg::Sensitivity spot{"model.spot",{.005}};
        const std::vector<bs::ModelParameters> bm(rows,{1.05f,.03f,.01f,.2f});
        pg::PriceGradientConfiguration bc{{spot,{"model.volatility",{.002}},
            {"product.strike",{.002}},{"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
        for (std::size_t k : {1U,4U,6U}) {
            auto selected=bc; selected.sensitivities.resize(k);
            const auto plan=bs::prepare_black_scholes_european_option_price_gradients(bm,products,PriceConstruction::Aligned,{},selected);
            measure("black_scholes",plan,bs::launch_black_scholes_european_option_price_gradients_cuda<OptionSide::call>,threads,width);
        }
        const std::vector<hs::ModelParameters> hm(rows,{1.05f,.03f,.01f,.04f,1.5f,.04f,.3f,-.7f});
        pg::PriceGradientConfiguration hc{{spot,
            {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"model.initial_variance",{.001,pg::BumpScale::absolute}},
            {"model.kappa",{.005}},{"model.theta",{.005}},{"model.gamma",{.005}},
            {"model.rho",{.002,pg::BumpScale::absolute}},{"product.strike",{.002}},
            {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
        for (std::size_t k : {1U,4U,10U}) {
            auto selected=hc; selected.sensitivities.resize(k);
            const auto plan=hs::prepare_heston_european_option_price_gradients(hm,products,PriceConstruction::Aligned,{},selected);
            measure("heston",plan,hs::launch_heston_european_option_price_gradients_cuda<OptionSide::call>,threads,width);
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
