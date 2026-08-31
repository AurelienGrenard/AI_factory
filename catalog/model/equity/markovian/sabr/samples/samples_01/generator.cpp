// Generated SABR conditional model-sample recipe.
#include "model/equity/markovian/sabr/sample.cuh"
#include "tools/sampling/generated/sabr_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::sabr;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668827815632961536ULL, 11668827816706703360ULL, 11668827817780445184ULL}
        )
    );
}
