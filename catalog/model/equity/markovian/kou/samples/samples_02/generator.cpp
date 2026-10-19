// Generated Kou unconditional model-sample recipe.
#include "model/equity/markovian/kou/sample.cuh"
#include "tools/sampling/generated/kou_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::kou;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668827403316101120ULL, 11668827404389842944ULL, 11668827405463584768ULL}
        )
    );
}
