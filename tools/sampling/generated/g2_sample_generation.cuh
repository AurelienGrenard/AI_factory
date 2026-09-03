// Generated G2 model-sample recipe composition.
#pragma once

#include "model/fixed_income/g2/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::g2 {

namespace model_binding = ai_factory::workbench::model::fixed_income::g2;
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
        const float mean_reversion_x = uniform({0.03f, 0.35f}, uniforms);
        const float mean_reversion_y = uniform({0.1f, 1.0f}, uniforms);
        const float volatility_x = uniform({0.0025f, 0.018f}, uniforms);
        const float volatility_y = uniform({0.0015f, 0.012f}, uniforms);
        const float correlation = uniform({-0.75f, 0.25f}, uniforms);
        const float initial_state_x = uniform({0.001f, 0.05f}, uniforms);
        const float initial_state_y = uniform({0.001f, 0.03f}, uniforms);
        if (!(true)) continue;
        parameters.push_back({{mean_reversion_x, volatility_x, mean_reversion_y, volatility_y, correlation}, {initial_state_x, initial_state_y}});
    }
    return parameters;
}

inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"mean_reversion_x", parameters.process.mean_reversion_x},
        {"volatility_x", parameters.process.volatility_x},
        {"mean_reversion_y", parameters.process.mean_reversion_y},
        {"volatility_y", parameters.process.volatility_y},
        {"correlation", parameters.process.correlation},
        {"initial_state_x", parameters.initial_state.state_x},
        {"initial_state_y", parameters.initial_state.state_y}
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
        "G2",
        "datasets/model/fixed_income/g2/samples/" + id + ".json",
        "catalog/model/fixed_income/g2/samples/" + id + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/g2/samples/" + id + ".json",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "exact finite-horizon transition",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals by parameter row"},
            {"latent_uniform_bounds", {
                {"mean_reversion_x", {0.03f, 0.35f}},
                {"mean_reversion_y", {0.1f, 1.0f}},
                {"volatility_x", {0.0025f, 0.018f}},
                {"volatility_y", {0.0015f, 0.012f}},
                {"correlation", {-0.75f, 0.25f}},
                {"initial_state_x", {0.001f, 0.05f}},
                {"initial_state_y", {0.001f, 0.03f}}
            }},
            {"acceptance", "true"}
        },
        {
            {"state_x", {{"description", "Terminal state_x."}, {"layout", "sample-major"}}},
            {"state_y", {{"description", "Terminal state_y."}, {"layout", "sample-major"}}}
        },
        {{"transition", "exact"}, {"delta_t", "maturity_days / 252"}, {"artificial_substeps", false}},
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
        {"state_x", "state_y"},
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
            throw std::invalid_argument("G2 sample output arity mismatch.");
        }
        model_binding::launch_g2_random_terminal_samples_cuda(
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

}  // namespace ai_factory::workbench::offline::sampling::g2
