// Generated Quadratic rough-Heston conditional model-sample recipe.
#include "model/equity/rough/quadratic_rough_heston/sample.cuh"
#include "tools/sampling/generated/quadratic_rough_heston_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::quadratic_rough_heston;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668828498532761600ULL, 11668828499606503424ULL, 11668828500680245248ULL}
        )
    );
}
