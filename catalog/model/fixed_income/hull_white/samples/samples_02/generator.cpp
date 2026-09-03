// Generated Hull-White unconditional model-sample recipe.
#include "model/fixed_income/hull_white/sample.cuh"
#include "tools/sampling/generated/hull_white_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::hull_white;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668829121303019520ULL, 11668829122376761344ULL, 11668829123450503168ULL}
        )
    );
}
