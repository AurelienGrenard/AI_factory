// Generated CIR conditional model-sample recipe.
#include "model/fixed_income/cir/sample.cuh"
#include "tools/sampling/generated/cir_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::cir;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668829048288575488ULL, 11668829049362317312ULL, 11668829050436059136ULL}
        )
    );
}
