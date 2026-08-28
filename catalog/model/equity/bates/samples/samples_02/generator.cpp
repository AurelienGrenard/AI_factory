// Generated Bates unconditional model-sample recipe.
#include "model/equity/markovian/bates/sample.cuh"
#include "tools/sampling/generated/bates_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::bates;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930001111ULL, 930001112ULL, 930001113ULL}
        )
    );
}
