// Generated Bates conditional model-sample recipe.
#include "model/equity/markovian/bates/sample.cuh"
#include "tools/sampling/generated/bates_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::bates;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668826767660941312ULL, 11668826768734683136ULL, 11668826769808424960ULL}
        )
    );
}
