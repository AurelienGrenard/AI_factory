// Generated CIR++ conditional model-sample recipe.
#include "model/fixed_income/cir_plus_plus/sample.cuh"
#include "tools/sampling/generated/cir_plus_plus_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::cir_plus_plus;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668829202907398144ULL, 11668829203981139968ULL, 11668829205054881792ULL}
        )
    );
}
