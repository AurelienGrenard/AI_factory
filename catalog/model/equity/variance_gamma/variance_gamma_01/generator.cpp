// Generate risk-neutral Variance-Gamma rows in ordered core and stress regimes.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

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
    float sigma_min;
    float sigma_max;
    float nu_min;
    float nu_max;
    float theta_min;
    float theta_max;
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
    std::uniform_real_distribution<float> sigma(
        bounds.sigma_min, bounds.sigma_max
    );
    std::uniform_real_distribution<float> nu(bounds.nu_min, bounds.nu_max);
    std::uniform_real_distribution<float> theta(
        bounds.theta_min, bounds.theta_max
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    std::size_t rejected = 0U;
    while (rows.size() < row_count) {
        const float candidate_sigma = sigma(generator);
        const float candidate_nu = nu(generator);
        const float candidate_theta = theta(generator);
        const float first_exponential_moment = 1.0f
            - candidate_theta * candidate_nu
            - 0.5f * candidate_sigma * candidate_sigma * candidate_nu;
        const float second_exponential_moment = 1.0f
            - 2.0f * candidate_theta * candidate_nu
            - 2.0f * candidate_sigma * candidate_sigma * candidate_nu;
        if (
            !(first_exponential_moment > 0.05f)
            || !(second_exponential_moment > 0.05f)
        ) {
            ++rejected;
            continue;
        }
        rows.push_back({
            {"spot", 1.0f},
            {"risk_free_rate", rate(generator)},
            {"dividend_yield", yield(generator)},
            {"sigma", candidate_sigma},
            {"nu", candidate_nu},
            {"theta", candidate_theta},
        });
    }
    return {
        std::move(rows),
        {
            {"method", "independent uniform proposals with rejection"},
            {"proposal_bounds", {
                {"spot", {1.0, 1.0}},
                {"risk_free_rate", {bounds.rate_min, bounds.rate_max}},
                {"dividend_yield", {bounds.yield_min, bounds.yield_max}},
                {"sigma", {bounds.sigma_min, bounds.sigma_max}},
                {"nu", {bounds.nu_min, bounds.nu_max}},
                {"theta", {bounds.theta_min, bounds.theta_max}},
            }},
            {
                "acceptance_condition",
                "1 - theta*nu - 0.5*sigma^2*nu > 0.05 and "
                "1 - 2*theta*nu - 2*sigma^2*nu > 0.05"
            },
            {"rejected_proposals", rejected},
            {
                "purpose",
                "guarantee both the martingale moment and a finite payoff "
                "variance for Monte Carlo validation"
            },
        },
    };
}

}  // namespace

int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/equity/variance_gamma/variance_gamma_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/equity/variance_gamma/variance_gamma_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/equity/"
        "variance_gamma/variance_gamma_01.json";
    constexpr std::uint64_t seed = 710000401ULL;

    GeneratedRows core = generate_regime(
        900U, seed,
        {0.001f, 0.08f, 0.0f, 0.06f, 0.08f, 0.45f,
         0.03f, 0.50f, -0.35f, 0.15f}
    );
    GeneratedRows stress = generate_regime(
        100U, seed + 1ULL,
        {-0.03f, 0.12f, 0.0f, 0.10f, 0.03f, 0.80f,
        0.05f, 1.50f, -0.80f, 0.60f}
    );
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_model_dataset(
        "variance_gamma_01",
        "Variance-Gamma",
        dataset_path,
        catalog_path,
        url,
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"sigma", "Brownian volatility in business time."},
            {"nu", "Variance rate of the unit-mean Gamma clock."},
            {"theta", "Drift of Brownian motion in business time."},
        },
        {
            {"log_return", "X_t = theta * G_t + sigma * W_(G_t)"},
            {"gamma_clock", "G_t ~ Gamma(shape=t/nu, scale=nu)"},
            {"martingale_correction", "omega = log(1 - theta*nu - sigma^2*nu/2) / nu"},
            {"spot", "S_t = S_0 exp((r-q+omega)t + X_t)"},
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
