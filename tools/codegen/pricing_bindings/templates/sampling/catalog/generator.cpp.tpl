// Generated $display $sample_kind model-sample recipe.
#include "model/$source_folder/sample.cuh"
#include "tools/sampling/generated/${model_name}_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::$model_name;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "$database_id",
            $parameter_count,
            $paths_per_parameter,
            {${parameter_seed}ULL, ${schedule_seed}ULL, ${dynamics_seed}ULL}
        )
    );
}
