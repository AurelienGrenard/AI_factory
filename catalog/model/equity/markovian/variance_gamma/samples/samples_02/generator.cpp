// Generated Variance-Gamma unconditional model-sample recipe.
#include "model/equity/markovian/variance_gamma/sample.cuh"
#include "tools/sampling/generated/variance_gamma_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::variance_gamma;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668828236539756544ULL, 11668828237613498368ULL, 11668828238687240192ULL}
        )
    );
}
