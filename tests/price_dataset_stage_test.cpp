#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/price_dataset.hpp"

#include <filesystem>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_price_dataset_stage_test";
    std::filesystem::remove_all(directory);
    const auto reference = [](const char* id) {
        return nlohmann::ordered_json{
            {"database_id", id},
            {"catalog", "catalog/test"},
            {"url", "https://datasets.ai-factory.example/test.json"},
        };
    };
    auto model = reference("model_01");
    model["model_family"] = "Test model";
    model["models"] = {{{"id", "000001"}, {"parameters", {{"x", 1.0}}}}};
    auto product = reference("product_01");
    product["product_family"] = "Test product";
    product["time_convention"] = {
        {"unit", "business_day"}, {"days_per_year", 252U}
    };
    product["products"] = {
        {{"id", "000001"}, {"parameters", {{"maturity", 252U}}}}
    };
    write_json_file(directory / "model.json", model);
    write_json_file(directory / "product.json", product);
    write_analytical_price_dataset(
        directory / "model.json",
        directory / "product.json",
        PriceConstruction::Aligned,
        std::vector<float>{1.25f},
        directory / "price.json",
        directory / "dataset.yaml",
        "https://datasets.ai-factory.example/test/price.json",
        "test formula",
        {{"block_count", 1U}},
        0.01,
        0.001
    );
    const auto result = read_json_file(directory / "price.json");
    if (result.at("row_count") != 1U
        || result.at("results").at(0).at("outputs").at("price") != 1.25f) {
        throw std::runtime_error("price artifact assembly failed");
    }
    std::filesystem::remove_all(directory);
}
