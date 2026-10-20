// Generated Rough-Heston model-sample recipe composition.
#pragma once

#include "model/equity/rough/rough_heston/sample.cuh"
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



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
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
        "datasets/model/equity/rough/rough_heston/samples/" + id + ".json",
        "catalog/model/equity/rough/rough_heston/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "seven-factor Markovian lift at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"spot", "initial_variance", "mean_reversion", "long_run_variance", "hurst_exponent", "rho"}},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"initial_variance", {0.01f, 0.12f}},
                {"mean_reversion", {0.5f, 4.0f}},
                {"long_run_variance", {0.01f, 0.15f}},
                {"hurst_exponent", {0.03f, 0.25f}},
                {"rho", {-0.95f, -0.25f}}
            }},
            {"acceptance", "true"},
            {"derived_parameters", {{"variance_drift", "variance_drift = mean_reversion * long_run_variance."}, {"volatility_of_variance", "After all latent uniform draws, draw volatility_of_variance uniformly on [max(sqrt(variance_drift / 5), 0.08), min(sqrt(12 * variance_drift), 0.8)] using the next Philox uniform."}}},
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
    const model_binding::SampleParameterSampler::Configuration bounds{{{1.0f, 1.0f}, {0.01f, 0.12f}, {0.5f, 4.0f}, {0.01f, 0.15f}, {0.03f, 0.25f}, {-0.95f, -0.25f}}};
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
            throw std::invalid_argument("Rough-Heston sample output arity mismatch.");
        }
        model_binding::launch_rough_heston_generated_random_terminal_samples_cuda<factor_count>(
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

}  // namespace ai_factory::workbench::offline::sampling::rough_heston
