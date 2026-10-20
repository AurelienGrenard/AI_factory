#include "correlation/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <nlohmann/json.hpp>

#include <vector>

int main() {
    using namespace ai_factory::workbench;
    const auto rough = datasets::read_json_file(
        "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json"
    );
    nlohmann::ordered_json rows = nlohmann::ordered_json::array();
    for (std::size_t i = 0U; i < 8U; ++i) {
        const double rho = rough.at("models").at(i)
            .at("parameters").at("rho").get<double>();
        const std::vector<double> dense{
            1.0, 0.08, 0.0,
            0.08, 1.0, rho,
            0.0, rho, 1.0
        };
        correlation::packed_cholesky(dense, 3U);
        rows.push_back({
            {"id", datasets::format_row_id(i)},
            {"factors", {"asset_0.spot", "asset_1.spot", "asset_1.vol"}},
            {"matrix", {{1.0, 0.08, 0.0},
                        {0.08, 1.0, rho},
                        {0.0, rho, 1.0}}}
        });
    }
    const auto path = "datasets/correlation/bs_rough_heston_matrices_01.json";
    datasets::write_json_file(path, {
        {"database_id", "bs_rough_heston_matrices_01"},
        {"catalog", "catalog/correlation/bs_rough_heston_matrices_01"},
        {"url", "https://datasets.ai-factory.example/v2/correlation/bs_rough_heston_matrices_01.json"},
        {"row_count", rows.size()},
        {"correlations", rows}
    });
    correlation::load_matrices(path);
    datasets::write_generation_receipt(
        "catalog/correlation/bs_rough_heston_matrices_01/generation.yaml",
        rows.size(), {{"mode", "rough_rho_matched_reference"}}, 0.0, 0.0
    );
}
