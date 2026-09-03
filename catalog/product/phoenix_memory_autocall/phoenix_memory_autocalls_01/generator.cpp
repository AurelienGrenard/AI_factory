// Generate issuance terms for memory-coupon Phoenix autocalls.
#include "tools/datasets/autocall_generation.hpp"
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

// Generate the Phoenix-Memory-autocall dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/phoenix_memory_autocall/"
        "phoenix_memory_autocalls_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/phoenix_memory_autocall/"
        "phoenix_memory_autocalls_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/"
        "phoenix_memory_autocalls/phoenix_memory_autocalls_01.json";

    constexpr std::size_t core_row_count = 900U;
    constexpr std::size_t tail_row_count = 100U;
    constexpr std::uint64_t seed = 731000301ULL;
    GeneratedRows rows = autocall::generate_phoenix_rows(
        core_row_count, tail_row_count, seed
    );
    rows.construction["paired_terms"] =
        "Rows match phoenix_autocalls_01 term by term.";

    write_product_dataset(
        "phoenix_memory_autocalls_01",
        "Phoenix Memory Autocalls",
        dataset_path,
        catalog_path,
        url,
        {
            {"maturity", "Time from issuance to final redemption in business days."},
            {"observation_interval", "Business days between observation dates."},
            {"autocall_barrier", "Spot level triggering early redemption."},
            {"coupon_barrier", "Spot level releasing current and remembered coupons."},
            {
                "protection_barrier",
                "Final spot level protecting the normalized nominal."
            },
            {
                "annual_coupon_rate",
                "Annual coupon rate prorated over each observation interval."
            },
        },
        {
            {
                "coupon_memory",
                "Missed coupons are paid at the next successful coupon "
                "observation."
            },
            {
                "early_redemption",
                "Before maturity, S_t >= autocall_barrier pays nominal 1 "
                "plus all due coupons and terminates the note."
            },
            {
                "protected_redemption",
                "At maturity, S_T >= protection_barrier repays nominal 1."
            },
            {
                "unprotected_redemption",
                "At maturity, S_T < protection_barrier repays S_T because "
                "S_0 = 1."
            },
            {"final_coupon", "Remembered coupons are lost unless S_T >= coupon_barrier."},
            {
                "protection_monitoring",
                "European: protection_barrier is observed only at maturity."
            },
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
