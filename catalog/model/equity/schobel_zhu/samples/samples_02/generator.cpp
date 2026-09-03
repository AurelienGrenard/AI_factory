// Generated Schobel-Zhu unconditional model-sample recipe.
#include "model/equity/markovian/schobel_zhu/sample.cuh"
#include "tools/sampling/generated/schobel_zhu_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::schobel_zhu;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {930010111ULL, 930010112ULL, 930010113ULL}
        )
    );
}
