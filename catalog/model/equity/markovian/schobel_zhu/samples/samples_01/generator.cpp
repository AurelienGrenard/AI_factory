// Generated Schobel-Zhu conditional model-sample recipe.
#include "model/equity/markovian/schobel_zhu/sample.cuh"
#include "tools/sampling/generated/schobel_zhu_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::schobel_zhu;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668827957366882304ULL, 11668827958440624128ULL, 11668827959514365952ULL}
        )
    );
}
