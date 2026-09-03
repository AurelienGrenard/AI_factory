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
            {930002101ULL, 930002102ULL, 930002103ULL}
        )
    );
}
