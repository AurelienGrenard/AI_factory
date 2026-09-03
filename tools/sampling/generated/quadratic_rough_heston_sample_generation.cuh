// Generated Quadratic rough-Heston model-sample recipe composition.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/sample.cuh"
#include "model/equity/rough/quadratic_rough_heston/numerics.hpp"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::quadratic_rough_heston {

namespace model_binding = ai_factory::workbench::model::equity::quadratic_rough_heston;
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
        const float initial_feedback = uniform({0.03f, 0.2f}, uniforms);
        const float quadratic_scale = uniform({0.1f, 0.8f}, uniforms);
        const float quadratic_shift = uniform({0.02f, 0.2f}, uniforms);
        const float variance_floor = uniform({0.0005f, 0.02f}, uniforms);
        const float feedback_rate = uniform({0.3f, 3.0f}, uniforms);
        const float feedback_volatility = uniform({0.3f, 2.0f}, uniforms);
        const float hurst_exponent = uniform({0.01f, 0.2f}, uniforms);
        if (!(true)) continue;
        parameters.push_back({spot, risk_free_rate, dividend_yield, initial_feedback, quadratic_scale, quadratic_shift, variance_floor, feedback_rate, feedback_volatility, hurst_exponent});
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
        {"initial_feedback", parameters.initial_feedback},
        {"quadratic_scale", parameters.quadratic_scale},
        {"quadratic_shift", parameters.quadratic_shift},
        {"variance_floor", parameters.variance_floor},
        {"feedback_rate", parameters.feedback_rate},
        {"feedback_volatility", parameters.feedback_volatility},
        {"hurst_exponent", parameters.hurst_exponent}
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
        "Quadratic rough-Heston",
        "datasets/model/equity/quadratic_rough_heston/samples/" + id + ".json",
        "catalog/model/equity/quadratic_rough_heston/samples/" + id + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/quadratic_rough_heston/samples/" + id + ".json",
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
                {"initial_feedback", {0.03f, 0.2f}},
                {"quadratic_scale", {0.1f, 0.8f}},
                {"quadratic_shift", {0.02f, 0.2f}},
                {"variance_floor", {0.0005f, 0.02f}},
                {"feedback_rate", {0.3f, 3.0f}},
                {"feedback_volatility", {0.3f, 2.0f}},
                {"hurst_exponent", {0.01f, 0.2f}}
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
            throw std::invalid_argument("Quadratic rough-Heston sample output arity mismatch.");
        }
        model_binding::launch_quadratic_rough_heston_random_terminal_samples_cuda<factor_count>(
            device_parameters, parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days, sample_offset,
            launch_sample_count, threads_per_block, block_count, schedule_seed, dynamics_seed,
            device_maturity_days,
            outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::quadratic_rough_heston
