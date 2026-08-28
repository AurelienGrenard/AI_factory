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
            {930003111ULL, 930003112ULL, 930003113ULL}
        )
    );
}
