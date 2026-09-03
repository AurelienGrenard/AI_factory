// Generated Black-Scholes conditional model-sample recipe.
#include "model/equity/markovian/black_scholes/sample.cuh"
#include "tools/sampling/generated/black_scholes_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::black_scholes;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668826840675385344ULL, 11668826841749127168ULL, 11668826842822868992ULL}
        )
    );
}
