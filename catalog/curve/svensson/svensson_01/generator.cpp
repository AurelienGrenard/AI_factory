// Generate Svensson curves from interpretable forward-rate levels.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"
#include "tools/datasets/svensson_generation.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace {

using ai_factory::workbench::datasets::svensson::SamplingRange;

constexpr std::size_t kRowCount = 1'000U;
constexpr std::uint64_t kSeed = 720000301ULL;
constexpr SamplingRange kLongForward{0.005f, 0.06f};
constexpr SamplingRange kShortForward{0.001f, 0.08f};
constexpr SamplingRange kFirstMediumForward{0.001f, 0.08f};
constexpr SamplingRange kSecondMediumForward{0.001f, 0.08f};
constexpr SamplingRange kTau1{0.5f, 3.0f};
constexpr SamplingRange kTau2{4.0f, 12.0f};
constexpr SamplingRange kAcceptedForward{0.001f, 0.10f};
constexpr SamplingRange kAcceptedCurvature{-0.20f, 0.20f};

}  // namespace

// Generate the Svensson dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/curve/svensson/svensson_01.json";
    const std::filesystem::path catalog_path =
        "catalog/curve/svensson/svensson_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/curve/"
        "svensson/svensson_01.json";

    const GeneratedRows rows = svensson::generate_rows(
        kRowCount,
        kSeed,
        {
            kLongForward,
            kShortForward,
            kFirstMediumForward,
            kSecondMediumForward,
            kTau1,
            kTau2,
            kAcceptedForward,
            kAcceptedCurvature,
        }
    );

    write_curve_dataset(
        "svensson_01",
        "Svensson",
        dataset_path,
        catalog_path,
        url,
        {
            {"beta0", "Long-maturity continuously compounded rate."},
            {"beta1", "Short-end slope loading."},
            {"beta2", "First medium-term curvature loading."},
            {"beta3", "Second medium-term curvature loading."},
            {"tau1", "Positive first decay time in years."},
            {"tau2", "Positive second decay time in years."},
        },
        {
            {
                "zero_rate",
                "z(0,T) = beta0 + beta1 L(T/tau1) "
                "+ beta2 C(T/tau1) + beta3 C(T/tau2)"
            },
            {"loading", "L(x) = (1 - exp(-x)) / x, C(x) = L(x) - exp(-x)"},
            {"discount_factor", "P(0,T) = exp(-T z(0,T))"},
            {
                "instantaneous_forward",
                "f(0,T) = beta0 + exp(-x1)(beta1 + beta2 x1) "
                "+ beta3 x2 exp(-x2)"
            },
            {"time_unit", "years"},
            {"rate_convention", "continuously compounded"},
        },
        rows
    );
    validate_curve_dataset_file(dataset_path);
}
