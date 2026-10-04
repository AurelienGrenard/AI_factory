// Generated Rough-SABR conditional model-sample recipe.
#include "model/equity/rough/rough_sabr/sample.cuh"
#include "tools/sampling/generated/rough_sabr_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_sabr;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668828897964720128ULL, 11668828899038461952ULL, 11668828900112203776ULL}
        )
    );
}
