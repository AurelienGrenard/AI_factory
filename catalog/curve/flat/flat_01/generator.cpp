// Generate flat continuously compounded curves over core and stress ranges.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace {

constexpr std::size_t kCoreRowCount = 900U;
constexpr std::size_t kStressRowCount = 100U;
constexpr std::uint64_t kCoreSeed = 720000101ULL;
constexpr std::uint64_t kStressSeed = 720000102ULL;

}  // namespace

// Generate the flat-curve dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/curve/flat/flat_01.json";
    const std::filesystem::path catalog_path =
        "catalog/curve/flat/flat_01/generation.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/curve/flat/flat_01.json";

    GeneratedRows rows = core_stress_rows(
        uniform_rows(kCoreRowCount, kCoreSeed, {
            {"rate", 0.001f, 0.08f},
        }),
        uniform_rows(kStressRowCount, kStressSeed, {
            {"rate", -0.02f, 0.12f},
        })
    );

    write_curve_dataset(
        "flat_01",
        "Flat",
        dataset_path,
        catalog_path,
        url,
        {
            {"rate", "Constant continuously compounded zero and forward rate."},
        },
        {
            {"zero_rate", "z(0,T) = rate"},
            {"discount_factor", "P(0,T) = exp(-rate T)"},
            {"instantaneous_forward", "f(0,T) = rate"},
            {"forward_derivative", "d_T f(0,T) = 0"},
            {"time_unit", "years"},
            {"rate_convention", "continuously compounded"},
        },
        rows
    );
    validate_curve_dataset_file(dataset_path);
}
