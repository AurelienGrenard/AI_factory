// Generated Hull-White conditional model-sample recipe.
#include "model/fixed_income/hull_white/sample.cuh"
#include "tools/sampling/generated/hull_white_sample_generation.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    namespace sampling = offline::sampling::hull_white;
    return sampling::generate(
        argc,
        argv,
        sampling::recipe(
            "samples_01",
            12'000U,
            250U,
            {11668829117008052224ULL, 11668829118081794048ULL, 11668829119155535872ULL}
        )
    );
}
