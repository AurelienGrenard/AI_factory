// Generated Heston 3/2 model-sample recipe composition.
#pragma once

#include "model/equity/markovian/heston_3_2/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::heston_3_2 {

namespace model_binding = ai_factory::workbench::model::equity::heston_3_2;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
        {"initial_variance", parameters.initial_variance},
        {"mean_reversion", parameters.mean_reversion},
        {"long_run_variance", parameters.long_run_variance},
        {"volatility_of_variance", parameters.volatility_of_variance},
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
        "Heston 3/2",
        "datasets/model/equity/markovian/heston_3_2/samples/" + id + ".json",
        "catalog/model/equity/markovian/heston_3_2/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "fixed-step transition at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"spot", "initial_variance", "mean_reversion", "long_run_variance", "volatility_of_variance", "rho"}},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"initial_variance", {0.01f, 0.09f}},
                {"mean_reversion", {5.0f, 40.0f}},
                {"long_run_variance", {0.015f, 0.09f}},
                {"volatility_of_variance", {1.0f, 8.0f}},
                {"rho", {-0.95f, -0.1f}}
            }},
            {"acceptance", "true"},
            {"fixed_model_inputs", {{"risk_free_rate", 0.0f}, {"dividend_yield", 0.0f}}}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}},
            {"reciprocal_variance", {{"description", "Terminal reciprocal_variance."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const model_binding::SampleParameterSampler::Configuration bounds{{{1.0f, 1.0f}, {0.01f, 0.09f}, {5.0f, 40.0f}, {0.015f, 0.09f}, {1.0f, 8.0f}, {-0.95f, -0.1f}}};
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
        {"spot", "reciprocal_variance"},
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
        if (outputs.size() != 2U) {
            throw std::invalid_argument("Heston 3/2 sample output arity mismatch.");
        }
        model_binding::launch_heston_3_2_generated_random_terminal_samples_cuda(
            device_parameters, bounds, parameter_count,
            paths_per_parameter, minimum_maturity_days,
            maximum_maturity_days, sample_offset, launch_sample_count,
            threads_per_block, block_count, parameter_seed, schedule_seed, dynamics_seed,
            device_maturity_days, outputs[0],
            outputs[1]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::heston_3_2
