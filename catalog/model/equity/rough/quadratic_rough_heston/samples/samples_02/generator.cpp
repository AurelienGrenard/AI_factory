// Generated Quadratic rough-Heston unconditional model-sample recipe.
#include "model/equity/rough/quadratic_rough_heston/sample.cuh"
#include "tools/sampling/generated/quadratic_rough_heston_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::quadratic_rough_heston;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668828502827728896ULL, 11668828503901470720ULL, 11668828504975212544ULL}
        )
    );
}
