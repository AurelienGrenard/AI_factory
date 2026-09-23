// Fixed-workload public-API comparison; keep allocations outside timed launches.
#include "model/equity/markovian/black_scholes/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/heston/product/european_option_price_gradients.cuh"
#include "model/equity/markovian/merton/product/european_option_price_gradients.cuh"
#include "tests/performance/benchmark_support.cuh"
#include <bit>
#include <cstdlib>
#include "benchmark_support.cuh"

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace bs = model::equity::black_scholes;
namespace hs = model::equity::heston;
namespace mt = model::equity::merton;
using namespace gradient_benchmark;

int main(int argc, char** argv) {
    try {
        const unsigned threads = argc > 1
            ? std::stoul(argv[1])
            : pg::kDefaultThreadsPerBlock;
        const std::size_t rows = argc > 2 ? std::stoul(argv[2]) : 64U;
        if (argc > 3 || rows == 0U) {
            throw std::invalid_argument("Usage: benchmark [THREADS [ROWS]]");
        }
        const bool profile_probe = std::getenv(
            "AI_FACTORY_PERFORMANCE_PROFILE_PROBE"
        ) != nullptr;
        const std::vector<product::EuropeanOptionParameters> products(rows,{1.f,126U});
        const pg::Sensitivity spot{"model.spot",{.005}};
        const std::vector<bs::ModelParameters> bm(rows,{1.05f,.03f,.01f,.2f});
        pg::PriceGradientConfiguration bc{{spot,{"model.volatility",{.002}},
            {"product.strike",{.002}},{"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
        const std::vector<std::size_t> black_scholes_widths = profile_probe
            ? std::vector<std::size_t>{6U}
            : std::vector<std::size_t>{1U, 4U, 6U};
        for (std::size_t k : black_scholes_widths) {
            auto selected=bc; selected.sensitivities.resize(k);
            const auto plan=bs::prepare_black_scholes_european_option_price_gradients(bm,products,PriceConstruction::Aligned,{},selected);
            measure("black_scholes",plan,bs::launch_black_scholes_european_option_price_gradients_cuda<OptionSide::call>,threads);
        }
        const std::vector<hs::ModelParameters> hm(rows,{1.05f,.03f,.01f,.04f,1.5f,.04f,.3f,-.7f});
        pg::PriceGradientConfiguration hc{{spot,
            {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"model.initial_variance",{.001,pg::BumpScale::absolute}},
            {"model.kappa",{.005}},{"model.theta",{.005}},{"model.gamma",{.005}},
            {"model.rho",{.002,pg::BumpScale::absolute}},{"product.strike",{.002}},
            {"product.maturity_years",{1.f/504.f,pg::BumpScale::absolute}}}};
        const std::vector<std::size_t> heston_widths = profile_probe
            ? std::vector<std::size_t>{10U}
            : std::vector<std::size_t>{1U, 4U, 10U};
        for (std::size_t k : heston_widths) {
            auto selected=hc; selected.sensitivities.resize(k);
            const auto plan=hs::prepare_heston_european_option_price_gradients(hm,products,PriceConstruction::Aligned,{},selected);
            measure("heston",plan,hs::launch_heston_european_option_price_gradients_cuda<OptionSide::call>,threads);
        }
        const pg::PriceGradientConfiguration heston_diagonal_configuration{{
            spot,
            {"model.initial_variance",{.001,pg::BumpScale::absolute}},
            {"model.kappa",{.005}},
            {"model.theta",{.005}},
            {"model.gamma",{.005}},
            {"model.rho",{.002,pg::BumpScale::absolute}},
            {"product.strike",{.002}},
        }};
        const auto heston_diagonal_plan =
            hs::prepare_heston_european_option_sensitivities(
                hm,
                products,
                PriceConstruction::Aligned,
                {},
                heston_diagonal_configuration,
                {pg::SensitivityOrders::first_and_second}
            );
        measure_diagonal(
            "heston",
            heston_diagonal_plan,
            hs::launch_heston_european_option_diagonal_sensitivities_cuda<
                OptionSide::call,
                pg::SensitivityOrders::first_and_second
            >,
            threads
        );
        const std::vector<mt::ModelParameters> mm(
            rows,
            {1.05f,.03f,.01f,.2f,.25f,-.08f,.20f}
        );
        pg::PriceGradientConfiguration mc{{spot,
            {"model.risk_free_rate",{.0001,pg::BumpScale::absolute}},
            {"model.dividend_yield",{.0001,pg::BumpScale::absolute}},
            {"model.volatility",{.002}},
            {"model.jump_log_mean",{.002,pg::BumpScale::absolute}},
            {"model.jump_log_volatility",{.002}},
            {"product.strike",{.002}}}};
        const std::vector<std::size_t> merton_widths = profile_probe
            ? std::vector<std::size_t>{7U}
            : std::vector<std::size_t>{1U, 4U, 7U};
        for (std::size_t k : merton_widths) {
            auto selected=mc; selected.sensitivities.resize(k);
            const auto plan=mt::prepare_merton_european_option_price_gradients(
                mm,products,PriceConstruction::Aligned,{},selected
            );
            measure("merton",plan,
                mt::launch_merton_european_option_price_gradients_cuda<OptionSide::call>,
                threads);
        }
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
