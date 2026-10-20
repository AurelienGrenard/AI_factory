// Generated Rough-Bergomi model-sample recipe composition.
#pragma once

#include "model/equity/rough/rough_bergomi/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::rough_bergomi {

namespace model_binding = ai_factory::workbench::model::equity::rough_bergomi;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;



inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
        {"xi_0", parameters.xi_0},
        {"eta", parameters.eta},
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
        "Rough-Bergomi",
        "datasets/model/equity/rough/rough_bergomi/samples/" + id + ".json",
        "catalog/model/equity/rough/rough_bergomi/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "block-cooperative hybrid FFT at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals with counter-local rejection per parameter row"},
            {"proposal_draw_order", {"spot", "xi_0", "eta", "hurst_exponent", "rho"}},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"xi_0", {0.04f, 0.04f}},
                {"eta", {0.5f, 3.0f}},
                {"hurst_exponent", {0.03f, 0.25f}},
                {"rho", {-0.95f, -0.3f}}
            }},
            {"acceptance", "true"},
            {"fixed_model_inputs", {{"risk_free_rate", 0.0f}, {"dividend_yield", 0.0f}}}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    const auto native_block = model_binding::rough_bergomi_sample_block_dimensions(value.maximum_maturity_days);    const model_binding::SampleParameterSampler::Configuration bounds{{{1.0f, 1.0f}, {0.04f, 0.04f}, {0.5f, 3.0f}, {0.03f, 0.25f}, {-0.95f, -0.3f}}};
    const std::uint64_t parameter_seed = value.seeds.parameters;

    return generate_device_model_sample_dataset<ModelParameters>(
        argc,
        argv,
        std::move(value),
        {
            ::ai_factory::workbench::offline::cuda_tuning::kSampleThreadsPerBlock,
            ::ai_factory::workbench::offline::cuda_tuning::kSampleBlockCountLimit,
            "volterra_samples", native_block
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
        unsigned int /* cuFFTDx fixes the block dimensions */,
        std::size_t block_count,
        std::uint64_t schedule_seed,
        std::uint64_t dynamics_seed,
        std::uint32_t* device_maturity_days,
        std::span<float*> outputs
    ) {
        if (outputs.size() != 1U) {
            throw std::invalid_argument("Rough-Bergomi sample output arity mismatch.");
        }
        model_binding::launch_rough_bergomi_generated_random_terminal_samples_cuda(
            device_parameters, bounds, parameter_count,
            paths_per_parameter, minimum_maturity_days,
            maximum_maturity_days, sample_offset, launch_sample_count,
            block_count, parameter_seed, schedule_seed, dynamics_seed,
            device_maturity_days, outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::rough_bergomi
