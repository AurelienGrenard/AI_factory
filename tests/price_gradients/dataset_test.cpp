// Native gradient artifact checks include one-sided stencils and fail-before-write behavior.
#include "tools/datasets/price_gradients/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace data = datasets::price_gradients;
    char pattern[] = "/tmp/ai_factory_gradients_artifact_XXXXXX";
    const auto* allocated = mkdtemp(pattern);
    if (!allocated) return 1;
    const std::filesystem::path directory(allocated);
    try {
        const auto require = [](bool value) { if (!value) throw std::runtime_error("Gradient artifact contract failed."); };
        nlohmann::ordered_json models{{"database_id","models"},{"catalog","catalog/test"},
            {"url","https://datasets.ai-factory.example/models.json"}, {"models",{{{"id","000001"}}}}};
        auto products = models;
        products.erase("models"); products["products"] = {{{"id","000001"}}};
        products["time_convention"] = {{"unit","business_day"},{"days_per_year",252}};
        datasets::write_json_file(directory/"models.json",models);
        datasets::write_json_file(directory/"products.json",products);
        data::Recipe recipe{directory/"models.json",directory/"products.json",directory/"gradients.json",directory/"dataset.yaml",
            "https://datasets.ai-factory.example/gradients.json","source/generator.cpp",PriceConstruction::Aligned,
            {{{"model.rho",{.125,pg::BumpScale::absolute}}}}, {}};
        data::Results result{{.1f},{.01f},{.2f},{.02f},
            {pg::prepare_stencil(1.f,{.125,pg::BumpScale::absolute},[](float p){return p>=-1 && p<=1;})},
            {{"paths_per_price",32},{"sensitivity_count",1},{"scenario_count",3},{"seed",719}}};
        data::write_dataset(recipe,result);
        const auto valid = datasets::read_json_file(recipe.dataset);
        require(valid["results"][0]["stencils"]["model.rho"]["kind"]=="backward");
        require(valid["validation"]["verified"]==false);
        require(valid["sensitivity"]["parameters"][0]["displacement"]==.125);
        result.gradients[0] = std::numeric_limits<float>::quiet_NaN();
        bool rejected = false;
        try { data::write_dataset(recipe,result); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && datasets::read_json_file(recipe.dataset)==valid);
        result.gradients[0] = .2f;
        result.execution["paths_per_price"] = 0;
        result.price_errors.clear(); result.gradient_errors.clear();
        data::write_dataset(recipe,result);
        require(!datasets::read_json_file(recipe.dataset)["results"][0]["outputs"].contains("gradient_standard_errors"));
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; std::filesystem::remove_all(directory); return 1;
    }
}
