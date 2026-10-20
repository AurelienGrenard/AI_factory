// Generated Quadratic rough-Heston model-sample recipe composition.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/sample.cuh"
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



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
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
        "datasets/model/equity/rough/quadratic_rough_heston/samples/" + id + ".json",
        "catalog/model/equity/rough/quadratic_rough_heston/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "seven-factor Markovian lift at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"spot", "initial_feedback", "quadratic_scale", "quadratic_shift", "variance_floor", "feedback_rate", "feedback_volatility", "hurst_exponent"}},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"initial_feedback", {0.02f, 0.15f}},
                {"quadratic_scale", {0.02f, 0.12f}},
                {"quadratic_shift", {0.02f, 0.15f}},
                {"variance_floor", {0.0005f, 0.015f}},
                {"feedback_rate", {0.1f, 0.8f}},
                {"feedback_volatility", {0.1f, 0.6f}},
                {"hurst_exponent", {0.05f, 0.2f}}
            }},
            {"acceptance", "true"},
            {"fixed_model_inputs", {{"risk_free_rate", 0.0f}, {"dividend_yield", 0.0f}}},
            {"kernel_preparation", {{"method", "positive bounded-L2 fit at sampled H"}, {"preparation_location", "GPU per sampled parameter row"}, {"horizon_years", 2.0f}, {"delta_t", "1 / 504"}}}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const model_binding::SampleParameterSampler::Configuration bounds{{{1.0f, 1.0f}, {0.02f, 0.15f}, {0.02f, 0.12f}, {0.02f, 0.15f}, {0.0005f, 0.015f}, {0.1f, 0.8f}, {0.1f, 0.6f}, {0.05f, 0.2f}}};
    const std::uint64_t parameter_seed = value.seeds.parameters;

    return generate_device_model_sample_dataset<ModelParameters>(
        argc,
        argv,
        std::move(value),
        {
            ::ai_factory::workbench::offline::cuda_tuning::kSampleThreadsPerBlock,
            ::ai_factory::workbench::offline::cuda_tuning::kSampleBlockCountLimit,
            "n_factor_samples"
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
            throw std::invalid_argument("Quadratic rough-Heston sample output arity mismatch.");
        }
        model_binding::launch_quadratic_rough_heston_generated_random_terminal_samples_cuda<factor_count>(
            device_parameters, bounds,
            parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days,
            sample_offset, launch_sample_count, threads_per_block,
            block_count, parameter_seed, schedule_seed, dynamics_seed,
            device_maturity_days, outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::quadratic_rough_heston
