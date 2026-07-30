// Generate reproducible Nelson-Siegel curves and their catalog artifacts.
#include "tools/datasets/dataset.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

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
    const std::filesystem::path generator_path =
        "catalog/curve/nelson_siegel/nelson_siegel_01/generator.cpp";

    constexpr std::uint64_t seed = 720000201ULL;
    const GeneratedRows rows = uniform_rows(1'000U, seed, {
        {"beta0", 0.0f, 0.08f},
        {"beta1", -0.06f, 0.04f},
        {"beta2", -0.08f, 0.08f},
        {"tau", 0.25f, 8.0f},
    });

    write_curve_dataset(
        "nelson_siegel_01",
        "Nelson-Siegel",
        dataset_path,
        catalog_path,
        url,
        generator_path,
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
}
