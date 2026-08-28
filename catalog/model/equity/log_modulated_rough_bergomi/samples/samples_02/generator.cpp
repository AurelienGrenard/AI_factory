// Generated Log-modulated rough-Bergomi unconditional model-sample recipe.
#include "model/equity/rough/log_modulated_rough_bergomi/sample.cuh"
#include "tools/sampling/generated/log_modulated_rough_bergomi_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::log_modulated_rough_bergomi;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930015111ULL, 930015112ULL, 930015113ULL}
        )
    );
}
