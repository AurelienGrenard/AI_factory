// Generated Black-Scholes unconditional model-sample recipe.
#include "model/equity/markovian/black_scholes/sample.cuh"
#include "tools/sampling/generated/black_scholes_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::black_scholes;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668826844970352640ULL, 11668826846044094464ULL, 11668826847117836288ULL}
        )
    );
}
