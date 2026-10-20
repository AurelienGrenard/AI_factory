// Generated Kou model-sample recipe composition.
#pragma once

#include "model/equity/markovian/kou/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::kou {

namespace model_binding = ai_factory::workbench::model::equity::kou;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
        {"volatility", parameters.volatility},
        {"jump_intensity", parameters.jump_intensity},
        {"up_probability", parameters.up_probability},
        {"positive_jump_rate", parameters.positive_jump_rate},
        {"negative_jump_rate", parameters.negative_jump_rate}
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
        "Kou",
        "datasets/model/equity/markovian/kou/samples/" + id + ".json",
        "catalog/model/equity/markovian/kou/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "exact finite-horizon transition",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"spot", "volatility", "jump_intensity", "up_probability", "positive_jump_rate", "negative_jump_rate"}},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"volatility", {0.08f, 0.45f}},
                {"jump_intensity", {0.02f, 1.0f}},
                {"up_probability", {0.2f, 0.7f}},
                {"positive_jump_rate", {3.0f, 20.0f}},
                {"negative_jump_rate", {2.0f, 20.0f}}
            }},
            {"acceptance", "true"},
            {"fixed_model_inputs", {{"risk_free_rate", 0.0f}, {"dividend_yield", 0.0f}}}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}}
        },
        {{"transition", "exact"}, {"delta_t", "maturity_days / 252"}, {"artificial_substeps", false}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const model_binding::SampleParameterSampler::Configuration bounds{{{1.0f, 1.0f}, {0.08f, 0.45f}, {0.02f, 1.0f}, {0.2f, 0.7f}, {3.0f, 20.0f}, {2.0f, 20.0f}}};
    const std::uint64_t parameter_seed = value.seeds.parameters;

    return generate_device_model_sample_dataset<ModelParameters>(
        argc,
        argv,
        std::move(value),
        {
            ::ai_factory::workbench::offline::cuda_tuning::kSampleThreadsPerBlock,
            ::ai_factory::workbench::offline::cuda_tuning::kSampleBlockCountLimit,
            "markovian_samples"
        },
        {"spot"},
        parameter_json,
        [bounds, parameter_seed](
        auto* device_parameters,
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
            throw std::invalid_argument("Kou sample output arity mismatch.");
        }
        model_binding::launch_kou_generated_random_terminal_samples_cuda(
            device_parameters, bounds, parameter_count,
            paths_per_parameter, minimum_maturity_days,
            maximum_maturity_days, sample_offset, launch_sample_count,
            threads_per_block, block_count, parameter_seed, schedule_seed, dynamics_seed,
            device_maturity_days, outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::kou
