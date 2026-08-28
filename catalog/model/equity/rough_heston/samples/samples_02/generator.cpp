// Generated Rough-Heston unconditional model-sample recipe.
#include "model/equity/rough/rough_heston/sample.cuh"
#include "tools/sampling/generated/rough_heston_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_heston;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930017111ULL, 930017112ULL, 930017113ULL}
        )
    );
}
