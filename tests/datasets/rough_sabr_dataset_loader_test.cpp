// Host-only validation of the rough-SABR model dataset boundary.
#include "model/equity/rough/rough_sabr/dataset.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void write_document(
    const std::filesystem::path& path,
    const nlohmann::json& document
) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write rough-SABR test JSON");
    output << document;
}

}  // namespace

int main() {
    namespace rough_sabr =
        ai_factory::workbench::model::equity::rough_sabr;
    const std::filesystem::path path =
        std::filesystem::temp_directory_path()
        / "ai_factory_rough_sabr_loader_test.json";
    nlohmann::json document = {
        {"database_id", "rough_sabr_test_01"},
        {"model_family", "Rough SABR"},
        {"catalog", "catalog/model/equity/rough/rough_sabr/test"},
        {"url", "https://datasets.ai-factory.example/rough_sabr_test.json"},
        {"row_count", 1U},
        {"models", {{
            {"id", "rough_sabr_001"},
            {"parameters", {
                {"spot", 1.0},
                {"risk_free_rate", 0.02},
                {"dividend_yield", 0.01},
                {"xi_0", 0.04},
                {"eta", 1.2},
                {"hurst_exponent", 0.1},
                {"rho", -0.7},
                {"beta", 0.7},
            }},
        }}},
    };
    write_document(path, document);
    const auto models = rough_sabr::load_models(path);
    require(models.size() == 1U, "rough-SABR loader rejected a valid row");
    require(models[0].beta == 0.7f, "rough-SABR beta was not loaded");

    document["models"][0]["parameters"]["beta"] = 0.49;
    write_document(path, document);
    try {
        static_cast<void>(rough_sabr::load_models(path));
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        require(
            message.find("rough_sabr_001") != std::string::npos,
            "rough-SABR loader error does not name the row"
        );
        require(
            message.find("beta") != std::string::npos,
            "rough-SABR loader error does not name beta"
        );
        std::filesystem::remove(path);
        return 0;
    }
    std::filesystem::remove(path);
    throw std::runtime_error("rough-SABR loader accepted beta below 0.5");
}
