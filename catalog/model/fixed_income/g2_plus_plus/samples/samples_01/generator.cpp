// Generated G2++ conditional model-sample recipe.
#include "model/fixed_income/g2_plus_plus/sample.cuh"
#include "tools/sampling/generated/g2_plus_plus_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::g2_plus_plus;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668829091238248448ULL, 11668829092311990272ULL, 11668829093385732096ULL}
        )
    );
}
