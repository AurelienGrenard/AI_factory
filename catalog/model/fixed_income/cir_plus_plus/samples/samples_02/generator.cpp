// Generated CIR++ unconditional model-sample recipe.
#include "model/fixed_income/cir_plus_plus/sample.cuh"
#include "tools/sampling/generated/cir_plus_plus_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::cir_plus_plus;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668829207202365440ULL, 11668829208276107264ULL, 11668829209349849088ULL}
        )
    );
}
