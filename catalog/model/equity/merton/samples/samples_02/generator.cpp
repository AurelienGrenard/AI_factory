// Generated Merton unconditional model-sample recipe.
#include "model/equity/markovian/merton/sample.cuh"
#include "tools/sampling/generated/merton_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::merton;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930007111ULL, 930007112ULL, 930007113ULL}
        )
    );
}
