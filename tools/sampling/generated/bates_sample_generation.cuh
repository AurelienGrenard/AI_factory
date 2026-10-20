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



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
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
        "catalog/model/equity/markovian/bates/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "fixed-step transition at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"spot", "initial_variance", "kappa", "theta", "rho", "jump_intensity", "jump_log_mean", "jump_log_volatility"}},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"initial_variance", {0.01f, 0.12f}},
                {"kappa", {0.5f, 4.0f}},
                {"theta", {0.01f, 0.15f}},
                {"rho", {-0.95f, -0.25f}},
                {"jump_intensity", {0.02f, 1.0f}},
                {"jump_log_mean", {-0.25f, 0.05f}},
                {"jump_log_volatility", {0.05f, 0.35f}}
            }},
            {"acceptance", "true"},
            {"derived_parameters", {{"gamma", "After all latent uniform draws, draw gamma uniformly on [max(sqrt(kappa * theta / 5), 0.1), min(sqrt(12 * kappa * theta), 0.8)] using the next Philox uniform."}}},
            {"fixed_model_inputs", {{"risk_free_rate", 0.0f}, {"dividend_yield", 0.0f}}}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}},
            {"variance", {{"description", "Terminal variance."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const model_binding::SampleParameterSampler::Configuration bounds{{{1.0f, 1.0f}, {0.01f, 0.12f}, {0.5f, 4.0f}, {0.01f, 0.15f}, {-0.95f, -0.25f}, {0.02f, 1.0f}, {-0.25f, 0.05f}, {0.05f, 0.35f}}};
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
        {"spot", "variance"},
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
            throw std::invalid_argument("Bates sample output arity mismatch.");
        }
        model_binding::launch_bates_generated_random_terminal_samples_cuda(
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

}  // namespace ai_factory::workbench::offline::sampling::bates
