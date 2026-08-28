// Generated $display model-sample recipe composition.
#pragma once

#include "model/$source_folder/sample.cuh"
$include_numerics#include "tools/sampling/host_philox.hpp"
#include "tools/sampling/model_sample_generation.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::offline::sampling::$model_name {

namespace model_binding = ai_factory::workbench::$namespace;
using ModelParameters = model_binding::ModelParameters;
inline constexpr std::size_t factor_count = 7U;

$parameter_factory

$parameter_json

inline datasets::ModelSampleRecipe recipe(
    const char* database_id,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    datasets::ModelSampleSeeds seeds
) {
    const std::string id(database_id);
    return {
        id,
        "$display",
        "datasets/model/$asset_class/$model_name/samples/" + id + ".json",
        "catalog/model/$asset_class/$model_name/samples/" + id + "/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/$asset_class/$model_name/samples/" + id + ".json",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "$numerical",
        {
            {"regime", "plausible core only"},
            {"distribution", "independent Philox uniform proposals by parameter row"},
            {"latent_uniform_bounds", $sample_bounds},
            {"acceptance", "$escaped_acceptance"},
        },
        {
            $output_metadata
        },
        $grid,
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
    return $generate_call(
        argc,
        argv,
        std::move(value),
        {
            ::ai_factory::workbench::offline::cuda_tuning::kSampleThreadsPerBlock,
            ::ai_factory::workbench::offline::cuda_tuning::kSampleBlockCountLimit,
            "$backend_samples"
        },
        {$output_names},
        generate_core_parameters,$prepare_argument
        parameter_json,
        $launch_lambda
    );
}

}  // namespace ai_factory::workbench::offline::sampling::$model_name
