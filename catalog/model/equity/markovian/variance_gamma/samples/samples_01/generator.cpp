// Generated Variance-Gamma conditional model-sample recipe.
#include "model/equity/markovian/variance_gamma/sample.cuh"
#include "tools/sampling/generated/variance_gamma_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::variance_gamma;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668828232244789248ULL, 11668828233318531072ULL, 11668828234392272896ULL}
        )
    );
}
