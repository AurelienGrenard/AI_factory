// Generated G2 unconditional model-sample recipe.
#include "model/fixed_income/g2/sample.cuh"
#include "tools/sampling/generated/g2_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::g2;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930020111ULL, 930020112ULL, 930020113ULL}
        )
    );
}
