// Generated Rough-Bergomi unconditional model-sample recipe.
#include "model/equity/rough/rough_bergomi/sample.cuh"
#include "tools/sampling/generated/rough_bergomi_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_bergomi;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930013111ULL, 930013112ULL, 930013113ULL}
        )
    );
}
