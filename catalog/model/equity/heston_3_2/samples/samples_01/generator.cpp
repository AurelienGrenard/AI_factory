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
            {930005101ULL, 930005102ULL, 930005103ULL}
        )
    );
}
