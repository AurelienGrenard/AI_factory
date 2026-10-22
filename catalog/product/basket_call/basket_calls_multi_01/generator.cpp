#include "tools/datasets/parameter_dataset.hpp"
#include "datasets/validation.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint32_t maturities[] = {
        21U,63U,126U,189U,252U,315U,378U,441U,504U,756U
    };
    constexpr float slope_min=-0.15f;
    constexpr float slope_max=0.15f;
    GeneratedRows rows;
    for (const auto maturity : maturities) {
        const float years=business_days_to_years(maturity);
        for (std::uint32_t strike_index=0U;strike_index<10U;
             ++strike_index) {
            const float slope=slope_min+(slope_max-slope_min)
                *static_cast<float>(strike_index)/9.0f;
            rows.rows.push_back({{"strike",std::exp(slope*years)},
                                  {"maturity",maturity}});
        }
    }
    rows.construction={{"method","ten maturities times ten exponential strikes"},
        {"maturities_business_days",maturities},
        {"log_moneyness_slope_min",slope_min},
        {"log_moneyness_slope_max",slope_max},
        {"core_only",true}};
    write_product_dataset(
        "basket_calls_multi_01","Arithmetic-mean basket call",
        "datasets/product/basket_call/basket_calls_multi_01.json",
        "catalog/product/basket_call/basket_calls_multi_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/product/basket_call/basket_calls_multi_01.json",
        {{"strike","Dimensionless strike on the terminal basket of initial-spot-normalized assets."},
          {"maturity","Business days from valuation; 252 days per year."}},
        {{"expression","max(mean_i(S_i(T)/S_i(0)) - strike, 0)"},
          {"notional","one normalized basket unit"},
          {"discounting","stochastic CIR short-rate integral"}},
        rows,
        {
            {"strike", {{"mode", "strike_on_mean_normalized_spot"},
                        {"unit", "dimensionless"},
                        {"underlying", "mean_i(S_i(T)/S_i(0))"}}}
        });
    ai_factory::workbench::datasets::validate_product_dataset_file(
        "datasets/product/basket_call/basket_calls_multi_01.json");
}
