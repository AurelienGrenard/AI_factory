// Fixed two-factor fixture. Random matrix generation is a separate concern.
#include "tools/datasets/artifact_io.hpp"
#include "correlation/dataset.hpp"

int main() {
    using namespace ai_factory::workbench;
    const std::vector<double> dense{1.0, 0.4, 0.4, 1.0};
    correlation::packed_cholesky(dense, 2U);
    datasets::write_json_file(
        "datasets/correlation/correlation_matrices_01.json",
        {
            {"database_id", "correlation_matrices_01"},
            {"catalog", "catalog/correlation/correlation_matrices_01"},
            {"url", "https://datasets.ai-factory.example/v1/correlation/correlation_matrices_01.json"},
            {"row_count", 1U},
            {"correlations", {
                {
                    {"id", "000001"},
                    {"factors", {"asset_0.spot", "asset_1.spot"}},
                    {"matrix", {{1.0, 0.4}, {0.4, 1.0}}}
                }
            }}
        }
    );
    correlation::load_matrices(
        "datasets/correlation/correlation_matrices_01.json"
    );
    datasets::write_generation_receipt(
        "catalog/correlation/correlation_matrices_01/generation.yaml",
        1U, {{"mode", "fixed_correlation_matrix"}}, 0.0, 0.0
    );
}
