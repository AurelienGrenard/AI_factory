// Generated Ornstein-Uhlenbeck conditional model-sample recipe.
#include "model/fixed_income/ornstein_uhlenbeck/sample.cuh"
#include "tools/sampling/generated/ornstein_uhlenbeck_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::ornstein_uhlenbeck;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668829134187921408ULL, 11668829135261663232ULL, 11668829136335405056ULL}
        )
    );
}
