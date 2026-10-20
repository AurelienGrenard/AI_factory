// Generated Normal-Inverse-Gaussian model-sample recipe composition.
#pragma once

#include "model/equity/markovian/normal_inverse_gaussian/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::normal_inverse_gaussian {

namespace model_binding = ai_factory::workbench::model::equity::normal_inverse_gaussian;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
        {"alpha", parameters.alpha},
        {"beta", parameters.beta},
        {"delta", parameters.delta}
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
        "Normal-Inverse-Gaussian",
        "datasets/model/equity/markovian/normal_inverse_gaussian/samples/" + id + ".json",
        "catalog/model/equity/markovian/normal_inverse_gaussian/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "exact finite-horizon transition",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"alpha", "skew_ratio", "target_volatility"}},
            {"latent_uniform_bounds", {
                {"alpha", {4.0f, 25.0f}},
                {"skew_ratio", {-0.75f, 0.05f}},
                {"target_volatility", {0.1f, 0.55f}}
            }},
            {"acceptance", "alpha > std::max(std::fabs(beta + 1.0f), std::fabs(beta + 2.0f)) + 0.05f"},
            {"derived_parameters", {{"spot", "Constant 1; no random draw."}, {"beta", "beta = skew_ratio * alpha."}, {"gamma", "gamma = sqrt(alpha * alpha - beta * beta); intermediate, not a published parameter."}, {"delta", "delta = target_volatility^2 * gamma^3 / alpha^2; deterministic reconstruction, before the declared acceptance test."}}},
            {"fixed_model_inputs", {{"risk_free_rate", 0.0f}, {"dividend_yield", 0.0f}}}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}}
        },
        {{"transition", "exact"}, {"delta_t", "maturity_days / 252"}, {"artificial_substeps", false}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const model_binding::SampleParameterSampler::Configuration bounds{{{4.0f, 25.0f}, {-0.75f, 0.05f}, {0.1f, 0.55f}}};
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
            throw std::invalid_argument("Normal-Inverse-Gaussian sample output arity mismatch.");
        }
        model_binding::launch_normal_inverse_gaussian_generated_random_terminal_samples_cuda(
            device_parameters, bounds, parameter_count,
            paths_per_parameter, minimum_maturity_days,
            maximum_maturity_days, sample_offset, launch_sample_count,
            threads_per_block, block_count, parameter_seed, schedule_seed, dynamics_seed,
            device_maturity_days, outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::normal_inverse_gaussian
