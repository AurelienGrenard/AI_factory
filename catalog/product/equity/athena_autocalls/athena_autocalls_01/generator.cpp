// Generate issuance terms for accumulated-gain Athena autocalls.
#include "tools/datasets/autocall_generation.hpp"
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

// Generate the Athena-autocall dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/athena_autocalls/athena_autocalls_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/athena_autocalls/"
        "athena_autocalls_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/athena_autocalls/"
        "athena_autocalls_01.json";

    constexpr std::size_t core_row_count = 900U;
    constexpr std::size_t tail_row_count = 100U;
    constexpr std::uint64_t seed = 731000401ULL;
    GeneratedRows rows = autocall::generate_athena_rows(
        core_row_count, tail_row_count, seed
    );

    write_product_dataset(
        "athena_autocalls_01",
        "Athena Autocalls",
        dataset_path,
        catalog_path,
        url,
        {
            {"maturity", "Time from issuance to final redemption in business days."},
            {"observation_interval", "Business days between observation dates."},
            {"autocall_barrier", "Spot level triggering early redemption."},
            {
                "protection_barrier",
                "Final spot level protecting the normalized nominal."
            },
            {"annual_coupon_rate", "Annual gain rate accumulated until redemption."},
        },
        {
            {
                "intermediate_payments",
                "None: the accumulated gain is paid only at redemption."
            },
            {
                "early_redemption",
                "At observation time t before maturity, S_t >= "
                "autocall_barrier pays 1 + annual_coupon_rate * t and "
                "terminates the note."
            },
            {
                "successful_maturity",
                "At maturity, S_T >= autocall_barrier pays 1 + "
                "annual_coupon_rate * maturity."
            },
            {
                "protected_redemption",
                "At maturity, protection_barrier <= S_T < autocall_barrier "
                "repays nominal 1 without gain."
            },
            {
                "unprotected_redemption",
                "At maturity, S_T < protection_barrier repays S_T because "
                "S_0 = 1."
            },
            {
                "protection_monitoring",
                "European: protection_barrier is observed only at maturity."
            },
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
