// Build CIR Bermudan-payer-swaption prices with Longstaff-Schwartz.
#include "model/fixed_income/cir/bermudan_swaption.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "tools/pricing/bermudan_swaption_price_generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace rates = model::fixed_income::cir;
    const std::filesystem::path model_path =
        "datasets/model/fixed_income/cir/parameters/cir_01.json";
    const std::filesystem::path product_path =
        "datasets/product/fixed_income/bermudan_swaptions/"
        "bermudan_swaptions_01.json";
    const auto models = rates::load_models(model_path);
    const auto products = product::load_bermudan_swaptions(product_path);
    constexpr std::size_t paths = 1U << 16U;
    constexpr std::uint64_t seed = 2'130'000'001ULL;
    datasets::generate_fixed_step_bermudan_swaption_prices(
        model_path, product_path, models, products,
        &rates::launch_cir_bermudan_swaption_cuda<SwaptionSide::payer>,
        1.0f / 504.0f, 2U,
        datasets::make_bermudan_swaption_generation_configuration(
            "cir", "payer", paths, seed,
            "Exact CIR endpoint with trapezoidal short-rate integral + Longstaff-Schwartz",
            "Hermite degree 3", "standardized short-rate factor", "1 / 504",
            {{"simulation_steps_per_day", 2U}}
        )
    );
}
