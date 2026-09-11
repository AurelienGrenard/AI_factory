// Generated $CASE_ID scaling probe; calls the published launcher with catalogue inputs.
#include "$HEADER"
#include "$MODEL_PREFIX/dataset.hpp"
#include "product/$PRODUCT/dataset.hpp"
$EXTRA_INCLUDE
#include "tests/performance/pricing_scaling_support.cuh"

int main(int argc, char** argv) {
    using namespace ai_factory::workbench;
    using namespace performance;
    using namespace performance::scaling;
    namespace model_binding = $MODEL_NAMESPACE;
    using Model = model_binding::ModelParameters;
    using Product = product::$PRODUCT_TYPE;
    try {
        const auto started = Clock::now();
        const auto jobs = read_jobs(argc, argv, $CLOSED_FORM);
        const auto input_paths = read_input_paths(argc, argv,
            {"$MODEL_DATASET", "$PRODUCT_DATASET", "$CURVE_DATASET"});
        const auto models = model_binding::load_models(input_paths.model);
        const auto products = product::$PRODUCT_LOADER(input_paths.product);
        validate_input_counts(models.size(), products.size(), jobs);
        DeviceBuffer device_models(models.size() * sizeof(Model), DeviceMemoryRole::persistent_input);
        DeviceBuffer device_products(products.size() * sizeof(Product), DeviceMemoryRole::persistent_input);
        DeviceBuffer device_prices(models.size() * sizeof(float), DeviceMemoryRole::output);
        DeviceBuffer device_errors(models.size() * sizeof(float), DeviceMemoryRole::output);
        copy_to_device(device_models, models);
        copy_to_device(device_products, products);
        constexpr std::uint64_t seed = ${SEED}ULL;
        $AUXILIARY_INPUTS
        const double preparation_ms = elapsed_ms(started);
        for (const auto& job : jobs) {
            benchmark(job, "$CASE_ID", $CLOSED_FORM, $VOLTERRA,
                preparation_ms, device_prices, device_errors,
                input_paths.model, input_paths.product, input_paths.curve, seed,
                "$TIME_STEP_DESCRIPTION",
                [&](std::size_t offset, std::size_t count, std::size_t blocks) {
                $RETURN$LAUNCHER<$TEMPLATE_ARGS>(
                    $LAUNCH_ARGS
                );
            });
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
