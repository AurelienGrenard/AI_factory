// Bounded SABR terminal workload: price, first order, and diagonal second order.
#include "model/equity/markovian/sabr/product/european_option_price_gradients.cuh"
#include "benchmark_support.cuh"

using namespace gradient_benchmark;
namespace sabr = model::equity::sabr;

int main() {
    try {
        const std::vector<sabr::ModelParameters> models(
            64U, {1.0f, .02f, .01f, .25f, .4f, -.3f, .7f}
        );
        const std::vector<product::EuropeanOptionParameters> products(
            64U, {1.0f, 126U}
        );
        const pg::PriceGradientConfiguration full{{
            {"model.spot", {.005}},
            {"model.risk_free_rate", {.0005, pg::BumpScale::absolute}},
            {"model.dividend_yield", {.0005, pg::BumpScale::absolute}},
            {"model.initial_volatility", {.005}},
            {"model.volatility_of_volatility", {.005}},
            {"model.rho", {.002, pg::BumpScale::absolute}},
            {"model.beta", {.002, pg::BumpScale::absolute}},
            {"product.strike", {.005}},
        }};
        for (unsigned int threads : {128U, 256U}) {
            for (unsigned int k : {0U, 1U, 8U}) {
                auto selected = full;
                selected.sensitivities.resize(k);
                const auto first = sabr::prepare_sabr_european_option_sensitivities(
                    models, products, PriceConstruction::Aligned, {}, selected,
                    {pg::SensitivityOrders::first}
                );
                measure(
                    "sabr_first", first,
                    sabr::launch_sabr_european_option_price_gradients_cuda<
                        OptionSide::call
                    >,
                    threads
                );
                if (k == 0U) continue;
                const auto diagonal = sabr::prepare_sabr_european_option_sensitivities(
                    models, products, PriceConstruction::Aligned, {}, selected,
                    {pg::SensitivityOrders::first_and_second}
                );
                measure_diagonal(
                    "sabr_first_and_second", diagonal,
                    sabr::launch_sabr_european_option_diagonal_sensitivities_cuda<
                        OptionSide::call,
                        pg::SensitivityOrders::first_and_second
                    >,
                    threads
                );
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
