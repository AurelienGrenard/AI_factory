// Verify model-sample dataset assembly and observable metadata serialization.
#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/sample_dataset.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_sample_dataset_stage_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::string progress_path = (directory / "progress.json").string();
    const std::string journal_path = (directory / "progress.jsonl").string();
    setenv("AI_FACTORY_GENERATION_PROGRESS", progress_path.c_str(), 1);
    setenv("AI_FACTORY_GENERATION_PROGRESS_LOG", journal_path.c_str(), 1);

    const std::vector<nlohmann::ordered_json> parameters = {
        {{"sigma", 0.2f}},
        {{"sigma", 0.3f}},
    };
    const std::vector<std::uint32_t> maturity_days = {63U, 64U, 503U, 504U};
    const std::vector<float> spots = {1.01f, 0.99f, 1.20f, 0.80f};
    const ModelSampleRecipe recipe{
        "samples_01",
        "Test model",
        directory / "samples_01.json",
        directory / "dataset.yaml",
        "https://datasets.ai-factory.example/test/samples_01.json",
        2U,
        2U,
        63U,
        504U,
        {101U, 102U, 103U},
        "test transition",
        {
            {"regime", "core"},
            {"bounds", {{"sigma", {0.1f, 0.4f}}}},
        },
        {{"spot", {{"description", "terminal spot"}}}},
        {{"transition", "exact"}},
    };
    write_model_sample_dataset(
        recipe,
        {
            2U,
            2U,
            true,
            0.01,
            0.001,
            {
                {"threads_per_block", 128U},
                {"block_count", 2U},
            },
        },
        [&](std::size_t index) { return parameters.at(index); },
        maturity_days,
        {{"spot", &spots}}
    );
    validate_model_sample_dataset_file(
        recipe.dataset_path,
        4U,
        63U,
        504U
    );
    const auto progress = read_json_file(progress_path);
    if (progress.at("state") != "complete"
        || progress.at("completed_samples") != 4U
        || progress.at("total_samples") != 4U
        || progress.at("phase") != "writing") {
        throw std::runtime_error("sample write progress did not finish");
    }
    std::ifstream journal(journal_path);
    std::size_t progress_records = 0U;
    for (std::string line; std::getline(journal, line);) {
        ++progress_records;
    }
    if (progress_records != 2U) {
        throw std::runtime_error("sample progress journal is incomplete");
    }
    const auto document = read_json_file(recipe.dataset_path);
    if (document.at("samples").at(0).at("parameters").at("sigma") != 0.2f
        || document.at("samples").at(1).at("parameters").at("sigma") != 0.2f
        || document.at("samples").at(2).at("parameters").at("sigma") != 0.3f) {
        throw std::runtime_error(
            "sample parameter packages are not parameter-major"
        );
    }
    std::ifstream yaml(recipe.catalog_path);
    const std::string yaml_text{
        std::istreambuf_iterator<char>(yaml),
        std::istreambuf_iterator<char>()
    };
    if (yaml_text.find("row_count: 4") == std::string::npos
        || yaml_text.find("days_per_year: 252") == std::string::npos
        || yaml_text.find("paths_per_parameter: 2") == std::string::npos) {
        throw std::runtime_error("sample catalog assembly failed");
    }
    std::filesystem::remove_all(directory);
    unsetenv("AI_FACTORY_GENERATION_PROGRESS");
    unsetenv("AI_FACTORY_GENERATION_PROGRESS_LOG");
}
