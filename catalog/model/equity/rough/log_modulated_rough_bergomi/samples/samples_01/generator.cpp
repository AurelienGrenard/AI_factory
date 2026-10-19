// Generated Log-modulated rough-Bergomi conditional model-sample recipe.
#include "model/equity/rough/log_modulated_rough_bergomi/sample.cuh"
#include "tools/sampling/generated/log_modulated_rough_bergomi_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::log_modulated_rough_bergomi;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668828365388775424ULL, 11668828366462517248ULL, 11668828367536259072ULL}
        )
    );
}
