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
            {930009101ULL, 930009102ULL, 930009103ULL}
        )
    );
}
