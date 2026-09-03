// Generated Heston conditional model-sample recipe.
#include "model/equity/markovian/heston/sample.cuh"
#include "tools/sampling/generated/heston_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::heston;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {930004101ULL, 930004102ULL, 930004103ULL}
        )
    );
}
