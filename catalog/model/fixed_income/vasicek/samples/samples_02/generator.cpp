// Generated Vasicek unconditional model-sample recipe.
#include "model/fixed_income/vasicek/sample.cuh"
#include "tools/sampling/generated/vasicek_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::vasicek;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_02",
            3'000'000U,
            1U,
            {11668829155662757888ULL, 11668829156736499712ULL, 11668829157810241536ULL}
        )
    );
}
