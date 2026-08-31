// Generate risk-neutral NIG rows from interpretable skew and volatility inputs.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <utility>

namespace {

using ai_factory::workbench::datasets::GeneratedRows;
using ai_factory::workbench::datasets::ParameterRow;

struct Bounds {
    float rate_min;
    float rate_max;
    float yield_min;
    float yield_max;
    float alpha_min;
    float alpha_max;
    float skew_ratio_min;
    float skew_ratio_max;
    float volatility_min;
    float volatility_max;
};

GeneratedRows generate_regime(
    std::size_t row_count,
    std::uint64_t seed,
    const Bounds& bounds
) {
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> rate(
        bounds.rate_min, bounds.rate_max
    );
    std::uniform_real_distribution<float> yield(
        bounds.yield_min, bounds.yield_max
    );
    std::uniform_real_distribution<float> alpha(
        bounds.alpha_min, bounds.alpha_max
    );
    std::uniform_real_distribution<float> skew_ratio(
        bounds.skew_ratio_min, bounds.skew_ratio_max
    );
    std::uniform_real_distribution<float> volatility(
        bounds.volatility_min, bounds.volatility_max
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    std::size_t rejected = 0U;
    while (rows.size() < row_count) {
        const float candidate_alpha = alpha(generator);
        const float candidate_beta =
            skew_ratio(generator) * candidate_alpha;
        const float required_tail = std::max(
            std::fabs(candidate_beta + 1.0f),
            std::fabs(candidate_beta + 2.0f)
        );
        if (candidate_alpha <= required_tail + 0.05f) {
            ++rejected;
            continue;
        }
        const float target_volatility = volatility(generator);
        const float gamma = std::sqrt(
            candidate_alpha * candidate_alpha
            - candidate_beta * candidate_beta
        );
        const float delta = target_volatility * target_volatility
            * gamma * gamma * gamma
            / (candidate_alpha * candidate_alpha);
        rows.push_back({
            {"spot", 1.0f},
            {"risk_free_rate", rate(generator)},
            {"dividend_yield", yield(generator)},
            {"alpha", candidate_alpha},
            {"beta", candidate_beta},
            {"delta", delta},
        });
    }
    return {
        std::move(rows),
        {
            {"method", "latent uniform sample, reconstruction, and rejection"},
            {"latent_uniform_bounds", {
                {"risk_free_rate", {bounds.rate_min, bounds.rate_max}},
                {"dividend_yield", {bounds.yield_min, bounds.yield_max}},
                {"alpha", {bounds.alpha_min, bounds.alpha_max}},
                {"skew_ratio_beta_over_alpha", {bounds.skew_ratio_min, bounds.skew_ratio_max}},
                {"target_annual_volatility", {bounds.volatility_min, bounds.volatility_max}},
            }},
            {"reconstruction", {
                {"beta", "skew_ratio_beta_over_alpha * alpha"},
                {"gamma", "sqrt(alpha^2 - beta^2)"},
                {"delta", "target_annual_volatility^2 * gamma^3 / alpha^2"},
            }},
            {
                "acceptance_condition",
                "alpha > max(abs(beta + 1), abs(beta + 2)) + 0.05"
            },
            {"rejected_proposals", rejected},
            {
                "purpose",
                "guarantee the first two exponential moments and sample "
                "volatility on an interpretable scale"
            },
        },
    };
}

}  // namespace

int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/equity/markovian/normal_inverse_gaussian/"
        "parameters/normal_inverse_gaussian_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/equity/markovian/normal_inverse_gaussian/"
        "parameters/normal_inverse_gaussian_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/equity/"
        "normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json";
    constexpr std::uint64_t seed = 710000501ULL;

    GeneratedRows core = generate_regime(
        900U, seed,
        {0.001f, 0.08f, 0.0f, 0.06f, 4.0f, 25.0f,
         -0.75f, 0.05f, 0.10f, 0.55f}
    );
    GeneratedRows stress = generate_regime(
        100U, seed + 1ULL,
        {-0.03f, 0.12f, 0.0f, 0.10f, 0.6f, 40.0f,
         -0.95f, 0.70f, 0.03f, 1.00f}
    );
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_model_dataset(
        "normal_inverse_gaussian_01",
        "Normal-Inverse-Gaussian",
        dataset_path,
        catalog_path,
        url,
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"alpha", "Positive NIG tail-steepness parameter."},
            {"beta", "NIG asymmetry parameter with abs(beta) < alpha."},
            {"delta", "Positive NIG scale parameter per year."},
        },
        {
            {"log_return", "X_t = beta * G_t + W_(G_t)"},
            {
                "inverse_gaussian_clock",
                "G_t ~ IG(delta*t/gamma, (delta*t)^2), "
                "gamma=sqrt(alpha^2-beta^2)"
            },
            {"martingale_correction", "omega = delta * (sqrt(alpha^2-(beta+1)^2) - gamma)"},
            {"spot", "S_t = S_0 exp((r-q+omega)t + X_t)"},
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
