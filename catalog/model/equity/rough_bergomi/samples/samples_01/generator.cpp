// Generated Rough-Bergomi conditional model-sample recipe.
#include "model/equity/rough/rough_bergomi/sample.cuh"
#include "tools/sampling/generated/rough_bergomi_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_bergomi;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {930013101ULL, 930013102ULL, 930013103ULL}
        )
    );
}
