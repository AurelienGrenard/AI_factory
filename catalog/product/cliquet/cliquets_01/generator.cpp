// Generate issuance terms for locally and globally capped Cliquets.
#include "tools/datasets/cliquet_generation.hpp"
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

// Generate the Cliquet dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/cliquet/cliquets_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/cliquet/cliquets_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/cliquets/"
        "cliquets_01.json";

    constexpr std::size_t core_row_count = 900U;
    constexpr std::size_t tail_row_count = 100U;
    constexpr std::uint64_t seed = 732000101ULL;
    GeneratedRows rows = cliquet::generate_rows(
        core_row_count, tail_row_count, seed
    );

    write_product_dataset(
        "cliquets_01",
        "Cliquets",
        dataset_path,
        catalog_path,
        url,
        {
            {"maturity", "Time from issuance to final payment in business days."},
            {
                "observation_interval",
                "Business days between consecutive return observations."
            },
            {
                "participation_rate",
                "Multiplier applied to every periodic spot return."
            },
            {"local_floor", "Lower bound applied to each periodic return."},
            {"local_cap", "Upper bound applied to each periodic return."},
            {"global_floor", "Lower bound applied to the accumulated return."},
            {"global_cap", "Upper bound applied to the accumulated return."},
        },
        {
            {
                "periodic_return",
                "R_i = S(t_i) / S(t_{i-1}) - 1, with S(t_0) at issuance."
            },
            {
                "local_return",
                "clamp(participation_rate * R_i, local_floor, local_cap)"
            },
            {
                "accumulated_return",
                "clamp(sum of local returns, global_floor, global_cap)"
            },
            {
                "maturity_payment",
                "Normalized nominal 1 plus the accumulated return."
            },
            {"intermediate_payments", "None."},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
