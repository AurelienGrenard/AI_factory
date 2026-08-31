// Generated Log-modulated rough-Bergomi model-sample recipe composition.
#pragma once

#include "model/equity/rough/log_modulated_rough_bergomi/sample.cuh"
#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::log_modulated_rough_bergomi {

namespace model_binding = ai_factory::workbench::model::equity::log_modulated_rough_bergomi;
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
        const float spot = uniform({1.0f, 1.0f}, uniforms);
        const float risk_free_rate = uniform({0.001f, 0.08f}, uniforms);
        const float dividend_yield = uniform({0.0f, 0.06f}, uniforms);
        const float xi_0 = uniform({0.01f, 0.09f}, uniforms);
        const float eta = uniform({0.5f, 3.0f}, uniforms);
        const float hurst_exponent = uniform({0.01f, 0.2f}, uniforms);
        const float rho = uniform({-0.95f, -0.2f}, uniforms);
        const float log_modulation_scale = uniform({0.03f, 0.3f}, uniforms);
        const float log_modulation_power = uniform({1.2f, 4.0f}, uniforms);
        if (!(true)) continue;
        parameters.push_back({spot, risk_free_rate, dividend_yield, xi_0, eta, hurst_exponent, rho, log_modulation_scale, log_modulation_power});
    }
    return parameters;
}

inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        {"spot", parameters.spot},
        {"risk_free_rate", parameters.risk_free_rate},
        {"dividend_yield", parameters.dividend_yield},
        {"xi_0", parameters.xi_0},
        {"eta", parameters.eta},
        {"hurst_exponent", parameters.hurst_exponent},
        {"rho", parameters.rho},
        {"log_modulation_scale", parameters.log_modulation_scale},
        {"log_modulation_power", parameters.log_modulation_power}
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
        "Log-modulated rough-Bergomi",
        "datasets/model/equity/rough/log_modulated_rough_bergomi/samples/" + id + ".json",
        "catalog/model/equity/rough/log_modulated_rough_bergomi/samples/" + id + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/samples/" + id + ".json",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "block-cooperative hybrid FFT at dt=1/504",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals by parameter row"},
            {"latent_uniform_bounds", {
                {"spot", {1.0f, 1.0f}},
                {"risk_free_rate", {0.001f, 0.08f}},
                {"dividend_yield", {0.0f, 0.06f}},
                {"xi_0", {0.01f, 0.09f}},
                {"eta", {0.5f, 3.0f}},
                {"hurst_exponent", {0.01f, 0.2f}},
                {"rho", {-0.95f, -0.2f}},
                {"log_modulation_scale", {0.03f, 0.3f}},
                {"log_modulation_power", {1.2f, 4.0f}}
            }},
            {"acceptance", "true"}
        },
        {
            {"spot", {{"description", "Terminal spot."}, {"layout", "sample-major"}}}
        },
        {{"transition", "fixed-step"}, {"delta_t", "1 / 504"}, {"simulation_steps_per_day", 2}},
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
            "volterra_samples"
        },
        {"spot"},
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
            throw std::invalid_argument("Log-modulated rough-Bergomi sample output arity mismatch.");
        }
        model_binding::launch_log_modulated_rough_bergomi_random_terminal_samples_cuda(
            device_parameters, parameter_count, paths_per_parameter,
            minimum_maturity_days, maximum_maturity_days, sample_offset,
            launch_sample_count, block_count, schedule_seed, dynamics_seed,
            device_maturity_days,
            outputs[0]
        );
    }
    );
}

}  // namespace ai_factory::workbench::offline::sampling::log_modulated_rough_bergomi
