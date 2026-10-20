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
        "datasets/model/$source_folder/samples/" + id + ".json",
        "catalog/model/$source_folder/samples/" + id + "/generation.yaml",
        parameter_count,
        paths_per_parameter,
        63U,
        504U,
        seeds,
        "$numerical",
        {
            {"regime", "plausible core only"},
            {"distribution", "$sampling_distribution"},
            {"proposal_draw_order", {$proposal_draw_order}},
            {"latent_uniform_bounds", $sample_bounds},
            {"acceptance", "$escaped_acceptance"}$derived_parameter_metadata$fixed_model_inputs_metadata$preparation_metadata
        },
        {
            $output_metadata
        },
        $grid,
    };
}

inline int generate(int argc, char** argv, datasets::ModelSampleRecipe value) {
$native_block_declaration$device_bounds_declaration
    return $generate_call(
        argc,
        argv,
        std::move(value),
        {
            ::ai_factory::workbench::offline::cuda_tuning::kSampleThreadsPerBlock,
            ::ai_factory::workbench::offline::cuda_tuning::kSampleBlockCountLimit,
            "$backend_samples"$native_block_argument
        },
        {$output_names},
        parameter_json,
        $launch_lambda
    );
}

}  // namespace ai_factory::workbench::offline::sampling::$model_name
