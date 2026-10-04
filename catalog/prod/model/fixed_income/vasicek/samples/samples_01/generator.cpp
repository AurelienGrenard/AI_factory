// Generated Vasicek conditional model-sample recipe.
#include "model/fixed_income/vasicek/sample.cuh"
#include "tools/sampling/generated/vasicek_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::vasicek;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668829151367790592ULL, 11668829152441532416ULL, 11668829153515274240ULL}
        )
    );
}
