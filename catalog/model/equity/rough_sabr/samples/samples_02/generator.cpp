// Generated Rough-SABR unconditional model-sample recipe.
#include "model/equity/rough/rough_sabr/sample.cuh"
#include "tools/sampling/generated/rough_sabr_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_sabr;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930014111ULL, 930014112ULL, 930014113ULL}
        )
    );
}
