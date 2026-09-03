// Generated Ornstein-Uhlenbeck unconditional model-sample recipe.
#include "model/fixed_income/ornstein_uhlenbeck/sample.cuh"
#include "tools/sampling/generated/ornstein_uhlenbeck_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::ornstein_uhlenbeck;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930023111ULL, 930023112ULL, 930023113ULL}
        )
    );
}
