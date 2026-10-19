// Generated Rough-SABR unconditional model-sample recipe.
#include "model/equity/rough/rough_sabr/sample.cuh"
#include "tools/sampling/generated/rough_sabr_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_sabr;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668828902259687424ULL, 11668828903333429248ULL, 11668828904407171072ULL}
        )
    );
}
