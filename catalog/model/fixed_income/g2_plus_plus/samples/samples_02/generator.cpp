// Generated G2++ unconditional model-sample recipe.
#include "model/fixed_income/g2_plus_plus/sample.cuh"
#include "tools/sampling/generated/g2_plus_plus_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::g2_plus_plus;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930021111ULL, 930021112ULL, 930021113ULL}
        )
    );
}
