// Generated Rough Stein-Stein unconditional model-sample recipe.
#include "model/equity/rough/rough_stein_stein/sample.cuh"
#include "tools/sampling/generated/rough_stein_stein_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_stein_stein;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668829035403673600ULL, 11668829036477415424ULL, 11668829037551157248ULL}
        )
    );
}
