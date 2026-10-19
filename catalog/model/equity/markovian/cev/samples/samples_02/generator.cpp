// Generated CEV unconditional model-sample recipe.
#include "model/equity/markovian/cev/sample.cuh"
#include "tools/sampling/generated/cev_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::cev;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668826986704273408ULL, 11668826987778015232ULL, 11668826988851757056ULL}
        )
    );
}
