// Verify model-parameter dataset assembly and metadata serialization.
#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_parameter_dataset_stage_test";
    std::filesystem::remove_all(directory);
    const GeneratedRows generated{{{{"sigma", 0.2f}}}, {{"method", "test"}}};
    write_model_dataset(
        "test_model_01",
        "Test model",
        directory / "model.json",
        directory / "generation.yaml",
        "https://datasets.ai-factory.example/test/model.json",
        {{"sigma", "volatility"}},
        {{"equation", "dS = sigma S dW"}},
        generated
    );
    const auto document = read_json_file(directory / "model.json");
    if (document.at("row_count") != 1U
        || document.at("models").at(0).at("id") != "000001") {
        throw std::runtime_error("parameter artifact assembly failed");
    }
    std::ifstream yaml(directory / "generation.yaml");
    const std::string receipt{std::istreambuf_iterator<char>(yaml), {}};
    if (receipt.find("status: \"complete\"") == std::string::npos
        || receipt.find("row_count: 1") == std::string::npos) {
        throw std::runtime_error("parameter generation receipt assembly failed");
    }
    std::filesystem::remove_all(directory);
}
