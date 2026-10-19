// Generated Normal-Inverse-Gaussian unconditional model-sample recipe.
#include "model/equity/markovian/normal_inverse_gaussian/sample.cuh"
#include "tools/sampling/generated/normal_inverse_gaussian_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::normal_inverse_gaussian;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668827686783942656ULL, 11668827687857684480ULL, 11668827688931426304ULL}
        )
    );
}
