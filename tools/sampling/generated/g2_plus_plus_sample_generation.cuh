// Generated G2++ model-sample recipe composition.
#pragma once

#include "model/fixed_income/g2_plus_plus/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::g2_plus_plus {

namespace model_binding = ai_factory::workbench::model::fixed_income::g2_plus_plus;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"mean_reversion_x", parameters.process.mean_reversion_x},
        {"volatility_x", parameters.process.volatility_x},
        {"mean_reversion_y", parameters.process.mean_reversion_y},
        {"volatility_y", parameters.process.volatility_y},
        {"correlation", parameters.process.correlation}
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
        "G2++",
        "datasets/model/fixed_income/g2_plus_plus/samples/" + id + ".json",
        "catalog/model/fixed_income/g2_plus_plus/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "exact finite-horizon transition",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"mean_reversion_x", "mean_reversion_gap", "stationary_standard_deviation_x", "stationary_standard_deviation_y", "correlation"}},
            {"latent_uniform_bounds", {
                {"mean_reversion_x", {0.03f, 0.35f}},
                {"mean_reversion_gap", {0.1f, 1.0f}},
                {"stationary_standard_deviation_x", {0.0025f, 0.018f}},
                {"stationary_standard_deviation_y", {0.0015f, 0.012f}},
                {"correlation", {-0.75f, -0.05f}}
            }},
            {"acceptance", "true"},
            {"derived_parameters", {{"mean_reversion_y", "mean_reversion_y = mean_reversion_x + mean_reversion_gap."}, {"volatility_x", "volatility_x = stationary_standard_deviation_x * sqrt(2 * mean_reversion_x)."}, {"volatility_y", "volatility_y = stationary_standard_deviation_y * sqrt(2 * mean_reversion_y)."}}}
        },
        {
            {"state_x", {{"description", "Terminal state_x."}, {"layout", "sample-major"}}},
            {"state_y", {{"description", "Terminal state_y."}, {"layout", "sample-major"}}}
        },
        {{"transition", "exact"}, {"delta_t", "maturity_days / 252"}, {"artificial_substeps", false}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const model_binding::SampleParameterSampler::Configuration bounds{{{0.03f, 0.35f}, {0.1f, 1.0f}, {0.0025f, 0.018f}, {0.0015f, 0.012f}, {-0.75f, -0.05f}}};
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
        {"state_x", "state_y"},
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
            throw std::invalid_argument("G2++ sample output arity mismatch.");
        }
        model_binding::launch_g2_plus_plus_generated_random_terminal_samples_cuda(
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

}  // namespace ai_factory::workbench::offline::sampling::g2_plus_plus
