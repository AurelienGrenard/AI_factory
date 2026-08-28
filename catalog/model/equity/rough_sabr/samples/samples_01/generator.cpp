// Generated Rough-SABR conditional model-sample recipe.
#include "model/equity/rough/rough_sabr/sample.cuh"
#include "tools/sampling/generated/rough_sabr_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_sabr;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {930014101ULL, 930014102ULL, 930014103ULL}
        )
    );
}
