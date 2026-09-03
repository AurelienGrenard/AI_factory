#include "tools/datasets/artifact_io.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

int main() {
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_artifact_io_stage_test";
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
