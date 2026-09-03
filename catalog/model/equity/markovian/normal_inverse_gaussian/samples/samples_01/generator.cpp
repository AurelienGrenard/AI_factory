// Generated Normal-Inverse-Gaussian conditional model-sample recipe.
#include "model/equity/markovian/normal_inverse_gaussian/sample.cuh"
#include "tools/sampling/generated/normal_inverse_gaussian_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::normal_inverse_gaussian;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668827682488975360ULL, 11668827683562717184ULL, 11668827684636459008ULL}
        )
    );
}
