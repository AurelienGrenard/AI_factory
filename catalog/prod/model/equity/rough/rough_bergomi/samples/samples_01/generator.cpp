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
            {11668828631676747776ULL, 11668828632750489600ULL, 11668828633824231424ULL}
        )
    );
}
