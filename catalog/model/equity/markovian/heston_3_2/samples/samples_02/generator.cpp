// Generated Heston 3/2 unconditional model-sample recipe.
#include "model/equity/markovian/heston_3_2/sample.cuh"
#include "tools/sampling/generated/heston_3_2_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::heston_3_2;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668827261582180352ULL, 11668827262655922176ULL, 11668827263729664000ULL}
        )
    );
}
