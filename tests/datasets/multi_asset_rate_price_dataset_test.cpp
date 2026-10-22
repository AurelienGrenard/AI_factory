#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    const auto directory=std::filesystem::temp_directory_path()
        / "ai_factory_multi_asset_rate_dataset_test";
    std::filesystem::create_directories(directory);
    const auto output=directory/"price.json";
    const auto receipt=directory/"generation.yaml";
    const auto correlation=directory/"correlation.json";
    datasets::write_json_file(correlation, {
        {"database_id", "synthetic_correlation_01"},
        {"correlations", {{{"id", "000001"}}}}
    });
    const auto equity=directory/"equity.json";
    const auto rate=directory/"rate.json";
    const auto product=directory/"product.json";
    const auto curve=directory/"curve.json";
    nlohmann::ordered_json models=nlohmann::ordered_json::array();
    for(std::uint32_t i=1U;i<=1000U;++i) {
        char id[7];
        std::snprintf(id,sizeof(id),"%06u",i);
        models.push_back({{"id",id}});
    }
    datasets::write_json_file(equity,{
        {"database_id","equity_01"},{"models",models}});
    datasets::write_json_file(rate,{
        {"database_id","g2_plus_plus_01"},{"models",models}});
    datasets::write_json_file(product,{
        {"database_id","product_01"},
        {"time_convention",{{"unit","business_day"},{"days_per_year",252}}},
        {"products",{{{"id","000001"}}}}});
    datasets::write_json_file(curve,{
        {"database_id","curve_01"},{"curves",models}});
    const datasets::MultiAssetPriceDatasetRecipe recipe{
        {equity,equity},product,correlation,curve,output,receipt,"",rate};
    using Expression=RowIndexExpression;
    const auto plan=compile_row_index_plan(
        Expression::cartesian({
            Expression::aligned({Expression::source(0U,1000U),
                                 Expression::source(1U,1000U),
                                 Expression::source(2U,1000U),
                                 Expression::source(5U,1000U)}),
            Expression::source(3U,1U),
            Expression::source(4U,1U)}),6U);
    std::vector<float> prices,errors;
    for(std::uint32_t j=0U;j<plan.result_count;++j) {
        prices.push_back(0.03f+0.00001f*j);
        errors.push_back(0.001f);
    }
    auto compact_recipe=recipe;
    compact_recipe.dataset_path=directory/"price_compact.json";
    datasets::write_multi_asset_price_dataset(
        compact_recipe,plan,prices,errors,128U,42U,
        {{"engine","cross_asset_mc"}},0.01,0.005);
    const auto compact_document=datasets::read_json_file(
        compact_recipe.dataset_path);
    if(compact_document.at("results").size()!=1000U
        ||compact_document.at("results").at(999).at("model_ids").size()!=2U
        ||compact_document.at("rate_model_dataset").at("id")
            !="g2_plus_plus_01"
        ||compact_document.at("results").at(0).at("rate_model_id")
            !="000001"
        ||compact_document.at("results").at(0).at("curve_id")
            !="000001")
        throw std::runtime_error("compact dataset provenance incomplete");
    bool rejected=false;
    try {
        auto invalid=plan;
        invalid.selectors[2].source_count=1000000U;
        datasets::write_multi_asset_price_dataset(
            recipe,invalid,prices,errors,128U,42U,
            {{"engine","cross_asset_mc"}},0.01,0.005);
    } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("invalid rate selector accepted");
    std::filesystem::remove_all(directory);
}
