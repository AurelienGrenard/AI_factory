// Generated G2 unconditional model-sample recipe.
#include "model/fixed_income/g2/sample.cuh"
#include "tools/sampling/generated/g2_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::g2;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668829069763411968ULL, 11668829070837153792ULL, 11668829071910895616ULL}
        )
    );
}
