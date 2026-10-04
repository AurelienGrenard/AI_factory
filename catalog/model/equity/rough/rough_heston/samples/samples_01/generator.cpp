// Generated Rough-Heston conditional model-sample recipe.
#include "model/equity/rough/rough_heston/sample.cuh"
#include "tools/sampling/generated/rough_heston_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_heston;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668828764820733952ULL, 11668828765894475776ULL, 11668828766968217600ULL}
        )
    );
}
