// Generated Heston unconditional model-sample recipe.
#include "model/equity/markovian/heston/sample.cuh"
#include "tools/sampling/generated/heston_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::heston;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668827128438194176ULL, 11668827129511936000ULL, 11668827130585677824ULL}
        )
    );
}
