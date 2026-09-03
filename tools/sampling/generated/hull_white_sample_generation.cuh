// Generated Hull-White model-sample recipe composition.
#pragma once

#include "model/fixed_income/hull_white/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::hull_white {

namespace model_binding = ai_factory::workbench::model::fixed_income::hull_white;
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
        const float mean_reversion = uniform({0.03f, 1.0f}, uniforms);
        const float stationary_volatility = uniform({0.0025f, 0.025f}, uniforms);
        const float volatility = stationary_volatility * std::sqrt(2.0f * mean_reversion);
        if (!(true)) continue;
        parameters.push_back({mean_reversion, volatility});
    }
    return parameters;
}

inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"mean_reversion", parameters.mean_reversion},
        {"volatility", parameters.volatility}
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
        "Hull-White",
        "datasets/model/fixed_income/hull_white/samples/" + id + ".json",
        "catalog/model/fixed_income/hull_white/samples/" + id + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/fixed_income/hull_white/samples/" + id + ".json",
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
                {"mean_reversion", {0.03f, 1.0f}},
                {"stationary_volatility", {0.0025f, 0.025f}}
            }},
            {"acceptance", "true"},
        },
        {
            {"state", {{"description", "Terminal state."}, {"layout", "sample-major"}}}
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
        {"state"},
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
        if (outputs.size() != 1U) {
            throw std::invalid_argument("Hull-White sample output arity mismatch.");
        }
        model_binding::launch_hull_white_random_terminal_samples_cuda(
            device_parameters, parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days, sample_offset,
            launch_sample_count, threads_per_block, block_count, schedule_seed, dynamics_seed,
            device_maturity_days,
            outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::hull_white
