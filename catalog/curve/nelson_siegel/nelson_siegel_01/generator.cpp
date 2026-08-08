// Generate Nelson-Siegel curves from interpretable forward-rate levels.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"
#include "tools/datasets/nelson_siegel_generation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace {

using ai_factory::workbench::datasets::nelson_siegel::SamplingRange;

constexpr std::size_t kRowCount = 1'000U;
constexpr std::uint64_t kSeed = 720000201ULL;
constexpr SamplingRange kLongForward{
    0.005f, 0.06f
};
constexpr SamplingRange kShortForward{
    0.001f, 0.08f
};
constexpr SamplingRange kMediumForward{
    0.001f, 0.08f
};
constexpr SamplingRange kTau{
    0.5f, 5.0f
};
constexpr SamplingRange kAcceptedForward{
    0.001f, 0.10f
};

}  // namespace

// Generate the Nelson-Siegel dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/curve/nelson_siegel/nelson_siegel_01.json";
    const std::filesystem::path catalog_path =
        "catalog/curve/nelson_siegel/nelson_siegel_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/curve/"
        "nelson_siegel/nelson_siegel_01.json";

    const GeneratedRows rows =
        nelson_siegel::generate_rows(
            kRowCount,
            kSeed,
            {
                kLongForward,
                kShortForward,
                kMediumForward,
                kTau,
                kAcceptedForward,
            }
        );

    write_curve_dataset(
        "nelson_siegel_01",
        "Nelson-Siegel",
        dataset_path,
        catalog_path,
        url,
        {
            {"beta0", "Long-maturity continuously compounded rate."},
            {"beta1", "Short-end slope loading."},
            {"beta2", "Medium-term curvature loading."},
            {"tau", "Positive decay time in years."},
        },
        {
            {
                "zero_rate",
                "z(0,T) = beta0 + beta1 L(T/tau) "
                "+ beta2 (L(T/tau) - exp(-T/tau))"
            },
            {"loading", "L(x) = (1 - exp(-x)) / x, with L(0) = 1"},
            {"discount_factor", "P(0,T) = exp(-T z(0,T))"},
            {
                "instantaneous_forward",
                "f(0,T) = beta0 + exp(-T/tau) "
                "(beta1 + beta2 T/tau)"
            },
            {
                "forward_derivative",
                "d_T f(0,T) = exp(-T/tau) "
                "(-beta1 + beta2 (1 - T/tau)) / tau"
            },
            {"time_unit", "years"},
            {"rate_convention", "continuously compounded"},
        },
        rows
    );
    validate_curve_dataset_file(dataset_path);
}
