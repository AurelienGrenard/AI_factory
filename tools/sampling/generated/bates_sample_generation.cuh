// Generated Bates model-sample recipe composition.
#pragma once

#include "model/equity/markovian/bates/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::bates {

namespace model_binding = ai_factory::workbench::model::equity::bates;
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
        const float kappa = uniform({0.5f, 4.0f}, uniforms);
        const float theta = uniform({0.01f, 0.15f}, uniforms);
        const float rho = uniform({-0.95f, -0.25f}, uniforms);
        const float jump_intensity = uniform({0.02f, 1.0f}, uniforms);
        const float jump_log_mean = uniform({-0.25f, 0.05f}, uniforms);
        const float jump_log_volatility = uniform({0.05f, 0.35f}, uniforms);
        const float gamma = uniform({std::max(std::sqrt(kappa * theta / 5.0f), 0.1f), std::min(std::sqrt(12.0f * kappa * theta), 0.8f)}, uniforms);
        if (!(true)) continue;
        parameters.push_back({spot, risk_free_rate, dividend_yield, initial_variance, kappa, theta, gamma, rho, jump_intensity, jump_log_mean, jump_log_volatility});
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
        {"kappa", parameters.kappa},
        {"theta", parameters.theta},
        {"gamma", parameters.gamma},
        {"rho", parameters.rho},
        {"jump_intensity", parameters.jump_intensity},
        {"jump_log_mean", parameters.jump_log_mean},
        {"jump_log_volatility", parameters.jump_log_volatility}
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
        "Bates",
        "datasets/model/equity/markovian/bates/samples/" + id + ".json",
        "catalog/model/equity/markovian/bates/samples/" + id + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/bates/samples/" + id + ".json",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "fixed-step transition at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals by parameter row"},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"risk_free_rate", {0.001f, 0.08f}},
                {"dividend_yield", {0.0f, 0.06f}},
                {"initial_variance", {0.01f, 0.12f}},
                {"kappa", {0.5f, 4.0f}},
                {"theta", {0.01f, 0.15f}},
                {"rho", {-0.95f, -0.25f}},
                {"jump_intensity", {0.02f, 1.0f}},
                {"jump_log_mean", {-0.25f, 0.05f}},
                {"jump_log_volatility", {0.05f, 0.35f}}
            }},
            {"acceptance", "true"}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}},
            {"variance", {{"description", "Terminal variance."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    return generate_model_sample_dataset<ModelParameters>(
        argc,
        argv,
        std::move(value),
        {
            ::ai_factory::workbench::offline::cuda_tuning::kSampleThreadsPerBlock,
            ::ai_factory::workbench::offline::cuda_tuning::kSampleBlockCountLimit,
            "markovian_samples"
        },
        {"spot", "variance"},
        generate_core_parameters,
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
        if (outputs.size() != 2U) {
            throw std::invalid_argument("Bates sample output arity mismatch.");
        }
        model_binding::launch_bates_random_terminal_samples_cuda(
            device_parameters, parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days, sample_offset,
            launch_sample_count, threads_per_block, block_count, schedule_seed, dynamics_seed,
            device_maturity_days,
            outputs[0],
            outputs[1]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::bates
