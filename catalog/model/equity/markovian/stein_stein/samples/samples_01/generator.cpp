// Generated Stein-Stein conditional model-sample recipe.
#include "model/equity/markovian/stein_stein/sample.cuh"
#include "tools/sampling/generated/stein_stein_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::stein_stein;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668828090510868480ULL, 11668828091584610304ULL, 11668828092658352128ULL}
        )
    );
}
