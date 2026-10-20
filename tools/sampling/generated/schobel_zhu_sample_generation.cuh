// Generated Schobel-Zhu model-sample recipe composition.
#pragma once

#include "model/equity/markovian/schobel_zhu/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::schobel_zhu {

namespace model_binding = ai_factory::workbench::model::equity::schobel_zhu;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
        {"initial_volatility", parameters.initial_volatility},
        {"mean_reversion", parameters.mean_reversion},
        {"long_run_volatility", parameters.long_run_volatility},
        {"volatility_of_volatility", parameters.volatility_of_volatility},
        {"correlation", parameters.correlation}
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
        "Schobel-Zhu",
        "datasets/model/equity/markovian/schobel_zhu/samples/" + id + ".json",
        "catalog/model/equity/markovian/schobel_zhu/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "fixed-step transition at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"spot", "initial_volatility", "mean_reversion", "long_run_volatility", "volatility_of_volatility", "correlation"}},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"initial_volatility", {0.08f, 0.45f}},
                {"mean_reversion", {0.3f, 5.0f}},
                {"long_run_volatility", {0.08f, 0.45f}},
                {"volatility_of_volatility", {0.03f, 0.6f}},
                {"correlation", {-0.9f, -0.1f}}
            }},
            {"acceptance", "true"},
            {"fixed_model_inputs", {{"risk_free_rate", 0.0f}, {"dividend_yield", 0.0f}}}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}},
            {"volatility", {{"description", "Terminal volatility."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const model_binding::SampleParameterSampler::Configuration bounds{{{1.0f, 1.0f}, {0.08f, 0.45f}, {0.3f, 5.0f}, {0.08f, 0.45f}, {0.03f, 0.6f}, {-0.9f, -0.1f}}};
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
        {"spot", "volatility"},
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
            throw std::invalid_argument("Schobel-Zhu sample output arity mismatch.");
        }
        model_binding::launch_schobel_zhu_generated_random_terminal_samples_cuda(
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

}  // namespace ai_factory::workbench::offline::sampling::schobel_zhu
