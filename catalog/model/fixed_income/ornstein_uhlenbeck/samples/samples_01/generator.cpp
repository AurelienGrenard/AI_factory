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
            {930023101ULL, 930023102ULL, 930023103ULL}
        )
    );
}
