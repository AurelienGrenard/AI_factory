// Generated Rough-Bergomi unconditional model-sample recipe.
#include "model/equity/rough/rough_bergomi/sample.cuh"
#include "tools/sampling/generated/rough_bergomi_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_bergomi;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668828635971715072ULL, 11668828637045456896ULL, 11668828638119198720ULL}
        )
    );
}
