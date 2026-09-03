// Generated CEV conditional model-sample recipe.
#include "model/equity/markovian/cev/sample.cuh"
#include "tools/sampling/generated/cev_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::cev;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668826982409306112ULL, 11668826983483047936ULL, 11668826984556789760ULL}
        )
    );
}
