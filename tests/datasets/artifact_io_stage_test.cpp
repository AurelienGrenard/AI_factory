// Verify JSON and YAML artifact serialization used by offline dataset stages.
#include "tools/datasets/artifact_io.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

int main() {
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_artifact_io_stage_test";
    for (const std::string prefix : {"equity/rough/rough_bergomi", "fixed_income/hull_white"}) {
        const auto metadata = price_validation_metadata(
            "datasets/model/" + prefix + "/prices/variant/example.json");
        if (metadata != nlohmann::ordered_json{
                {"status", "pending"}, {"verified", false},
                {"dataset", "validation/datasets/price/" + prefix + "/variant/example.json"}}) {
            throw std::runtime_error("Generation must publish pending validation and the canonical cache path");
        }
    }
    if (price_validation_metadata(directory / "example.json")
        != nlohmann::ordered_json{{"status", "pending"}, {"verified", false}}) {
        throw std::runtime_error("Temporary artifacts must not advertise an invented reference");
    }
    std::filesystem::remove_all(directory);
    write_json_file(directory / "document.json", {{"answer", 42}});
    write_catalog_yaml(directory / "catalog.yaml", {{"answer", 42}});
    if (read_json_file(directory / "document.json").at("answer") != 42) {
        throw std::runtime_error("JSON artifact round trip failed");
    }
    std::ifstream yaml(directory / "catalog.yaml");
    const std::string contents{
        std::istreambuf_iterator<char>(yaml),
        std::istreambuf_iterator<char>()
    };
    if (contents.find("answer: 42") == std::string::npos
        || format_row_id(0U) != "000001") {
        throw std::runtime_error("YAML artifact or row id is incorrect");
    }
    std::filesystem::remove_all(directory);
}
