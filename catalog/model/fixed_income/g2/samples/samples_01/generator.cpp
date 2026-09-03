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
            {930020101ULL, 930020102ULL, 930020103ULL}
        )
    );
}
