// Generated G2 conditional model-sample recipe.
#include "model/fixed_income/g2/sample.cuh"
#include "tools/sampling/generated/g2_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::g2;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668829065468444672ULL, 11668829066542186496ULL, 11668829067615928320ULL}
        )
    );
}
