// Generated CIR unconditional model-sample recipe.
#include "model/fixed_income/cir/sample.cuh"
#include "tools/sampling/generated/cir_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::cir;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668829052583542784ULL, 11668829053657284608ULL, 11668829054731026432ULL}
        )
    );
}
