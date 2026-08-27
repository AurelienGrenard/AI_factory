// Generate issuance terms for equity Range Accrual notes.
#include "tools/datasets/range_accrual_generation.hpp"
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

// Generate the Range Accrual dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/range_accruals/range_accruals_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/range_accruals/range_accruals_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/range_accruals/"
        "range_accruals_01.json";

    constexpr std::size_t core_row_count = 900U;
    constexpr std::size_t tail_row_count = 100U;
    constexpr std::uint64_t seed = 733000101ULL;
    GeneratedRows rows = range_accrual::generate_rows(
        core_row_count, tail_row_count, seed
    );

    write_product_dataset(
        "range_accruals_01",
        "Range Accruals",
        dataset_path,
        catalog_path,
        url,
        {
            {"maturity", "Time from issuance to final payment in business days."},
            {
                "observation_interval",
                "Business days between consecutive spot observations."
            },
            {"lower_barrier", "Lower normalized spot observation bound."},
            {"upper_barrier", "Upper normalized spot observation bound."},
            {"coupon_rate", "Annualized coupon rate accrued in range."},
        },
        {
            {
                "in_range_count",
                "Sum of 1{lower_barrier <= S(t_i) / S(0) <= upper_barrier}."
            },
            {
                "accrued_coupon",
                "coupon_rate * observation_interval * in_range_count"
            },
            {
                "maturity_payment",
                "Normalized nominal 1 plus the accrued coupon."
            },
            {"intermediate_payments", "None."},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
