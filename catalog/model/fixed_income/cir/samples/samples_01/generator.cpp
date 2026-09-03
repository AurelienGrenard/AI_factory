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
            {930019101ULL, 930019102ULL, 930019103ULL}
        )
    );
}
