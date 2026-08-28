// Generated Heston 3/2 unconditional model-sample recipe.
#include "model/equity/markovian/heston_3_2/sample.cuh"
#include "tools/sampling/generated/heston_3_2_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::heston_3_2;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930005111ULL, 930005112ULL, 930005113ULL}
        )
    );
}
