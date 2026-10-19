// Generated SABR unconditional model-sample recipe.
#include "model/equity/markovian/sabr/sample.cuh"
#include "tools/sampling/generated/sabr_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::sabr;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668827819927928832ULL, 11668827821001670656ULL, 11668827822075412480ULL}
        )
    );
}
