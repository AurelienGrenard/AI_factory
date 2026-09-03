// Generated Stein-Stein unconditional model-sample recipe.
#include "model/equity/markovian/stein_stein/sample.cuh"
#include "tools/sampling/generated/stein_stein_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::stein_stein;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668828094805835776ULL, 11668828095879577600ULL, 11668828096953319424ULL}
        )
    );
}
