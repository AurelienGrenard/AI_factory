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
            {930017101ULL, 930017102ULL, 930017103ULL}
        )
    );
}
