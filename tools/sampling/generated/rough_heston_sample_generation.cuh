// Generated Rough-Heston model-sample recipe composition.
#pragma once

#include "model/equity/rough/rough_heston/sample.cuh"
#include "model/equity/rough/rough_heston/markovian_n_factor_preparation.hpp"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::rough_heston {

namespace model_binding = ai_factory::workbench::model::equity::rough_heston;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;

inline std::vector<ModelParameters> generate_core_parameters(
    std::size_t parameter_count,
    std::uint64_t seed
) {
    std::vector<ModelParameters> parameters;
    parameters.reserve(parameter_count);
    std::size_t proposal = 0U;
    while (parameters.size() < parameter_count) {
        HostUniformSequence uniforms(seed, proposal++);
        const float spot = uniform({1.0f, 1.0f}, uniforms);
        const float risk_free_rate = uniform({0.001f, 0.08f}, uniforms);
        const float dividend_yield = uniform({0.0f, 0.06f}, uniforms);
        const float initial_variance = uniform({0.01f, 0.12f}, uniforms);
        const float mean_reversion = uniform({0.5f, 4.0f}, uniforms);
        const float long_run_variance = uniform({0.01f, 0.15f}, uniforms);
        const float hurst_exponent = uniform({0.03f, 0.25f}, uniforms);
        const float rho = uniform({-0.95f, -0.25f}, uniforms);
        const float variance_drift = mean_reversion * long_run_variance; const float volatility_of_variance = uniform({std::max(std::sqrt(variance_drift / 5.0f), 0.08f), std::min(std::sqrt(12.0f * variance_drift), 0.8f)}, uniforms);
        if (!(true)) continue;
        parameters.push_back({spot, risk_free_rate, dividend_yield, initial_variance, mean_reversion, variance_drift, volatility_of_variance, hurst_exponent, rho});
    }
    return parameters;
}

inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
        {"risk_free_rate", parameters.risk_free_rate},
        {"dividend_yield", parameters.dividend_yield},
        {"initial_variance", parameters.initial_variance},
        {"mean_reversion", parameters.mean_reversion},
        {"variance_drift", parameters.variance_drift},
        {"volatility_of_variance", parameters.volatility_of_variance},
        {"hurst_exponent", parameters.hurst_exponent},
        {"rho", parameters.rho}
    };
}

inline datasets::ModelSampleRecipe recipe(
    const char* database_id,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    datasets::ModelSampleSeeds seeds
) {
    const std::string id(database_id);
    return {
        id,
        "Rough-Heston",
        "datasets/model/equity/rough_heston/samples/" + id + ".json",
        "catalog/model/equity/rough_heston/samples/" + id + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough_heston/samples/" + id + ".json",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "seven-factor Markovian lift at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals by parameter row"},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"risk_free_rate", {0.001f, 0.08f}},
                {"dividend_yield", {0.0f, 0.06f}},
                {"initial_variance", {0.01f, 0.12f}},
                {"mean_reversion", {0.5f, 4.0f}},
                {"long_run_variance", {0.01f, 0.15f}},
                {"hurst_exponent", {0.03f, 0.25f}},
                {"rho", {-0.95f, -0.25f}}
            }},
            {"acceptance", "true"},
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    return generate_prepared_model_sample_dataset<ModelParameters, model_binding::PreparedDynamics<factor_count>>(
        argc,
        argv,
        std::move(value),
        {
            ::ai_factory::workbench::offline::cuda_tuning::kSampleThreadsPerBlock,
            ::ai_factory::workbench::offline::cuda_tuning::kSampleBlockCountLimit,
            "n_factor_samples"
        },
        {"spot"},
        generate_core_parameters,
        [](const std::vector<ModelParameters>& parameters,
           std::uint32_t maximum_maturity_days) {
            return model_binding::prepare_dynamics<factor_count>(
                parameters,
                static_cast<float>(maximum_maturity_days) / 252.0f,
                1.0f / 504.0f
            );
        },
        parameter_json,
        [](
        const auto* device_parameters,
        std::size_t parameter_count,
        std::size_t paths_per_parameter,
        std::uint32_t minimum_maturity_days,
        std::uint32_t maximum_maturity_days,
        std::size_t sample_offset,
        std::size_t launch_sample_count,
        unsigned int threads_per_block,
        std::size_t block_count,
        std::uint64_t schedule_seed,
        std::uint64_t dynamics_seed,
        std::uint32_t* device_maturity_days,
        std::span<float*> outputs
    ) {
        if (outputs.size() != 1U) {
            throw std::invalid_argument("Rough-Heston sample output arity mismatch.");
        }
        model_binding::launch_rough_heston_random_terminal_samples_cuda<factor_count>(
            device_parameters, parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days, sample_offset,
            launch_sample_count, threads_per_block, block_count, schedule_seed, dynamics_seed,
            device_maturity_days,
            outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::rough_heston
