// Generated Rough Stein-Stein conditional model-sample recipe.
#include "model/equity/rough/rough_stein_stein/sample.cuh"
#include "tools/sampling/generated/rough_stein_stein_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::rough_stein_stein;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {930016101ULL, 930016102ULL, 930016103ULL}
        )
    );
}
