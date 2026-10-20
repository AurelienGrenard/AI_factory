#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

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
    const std::filesystem::path equity=
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json";
    const std::filesystem::path rate=
        "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json";
    const datasets::MultiAssetPriceDatasetRecipe recipe{
        {equity,equity},
        "datasets/product/worst_of_call/worst_of_calls_01.json",
        "datasets/correlation/bs_rough_heston_matrices_01.json",
        "datasets/curve/flat/flat_01.json",output,receipt,
        "https://datasets.ai-factory.example/v2/multi_asset/worst_of_call/rate_test.json",
        rate};
    datasets::write_multi_asset_price_dataset(
        recipe,{0U,0U},{{0U,0U,0U,0U,1.0f}},
        {0.03f},{0.001f},128U,42U,
        {{"engine","cross_asset_mc"}},0.01,0.005,{0U});
    const auto document=datasets::read_json_file(output);
    if(document.at("rate_model_dataset").at("id")!="g2_plus_plus_01"
        || document.at("results").at(0).at("rate_model_id")!="000001"
        || document.at("results").at(0).at("curve_id")!="000001")
        throw std::runtime_error("missing short-rate dataset provenance");
    bool rejected=false;
    try {
        datasets::write_multi_asset_price_dataset(
            recipe,{0U,0U},{{0U,0U,0U,0U,1.0f}},
            {0.03f},{0.001f},128U,42U,
            {{"engine","cross_asset_mc"}},0.01,0.005,{1000000U});
    } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("invalid rate index accepted");
    std::filesystem::remove_all(directory);
}
