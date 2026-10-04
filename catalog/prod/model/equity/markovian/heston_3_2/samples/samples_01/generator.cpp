// Generated Heston 3/2 conditional model-sample recipe.
#include "model/equity/markovian/heston_3_2/sample.cuh"
#include "tools/sampling/generated/heston_3_2_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::heston_3_2;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668827257287213056ULL, 11668827258360954880ULL, 11668827259434696704ULL}
        )
    );
}
