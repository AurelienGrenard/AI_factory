// Generated Merton conditional model-sample recipe.
#include "model/equity/markovian/merton/sample.cuh"
#include "tools/sampling/generated/merton_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::merton;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {930007101ULL, 930007102ULL, 930007103ULL}
        )
    );
}
