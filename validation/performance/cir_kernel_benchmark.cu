// Compare the current and targeted no-inline CIR distribution implementations.
#include "validation/performance/benchmark_support.cuh"

#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/cir/product/european_swaption.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef AI_FACTORY_CIR_BENCHMARK_VARIANT
#define AI_FACTORY_CIR_BENCHMARK_VARIANT "current_inline"
#endif

int main() {
    using namespace ai_factory::workbench;
    namespace cir = model::fixed_income::cir;
    namespace product = ai_factory::workbench::product;
    using performance::DeviceBuffer;
    // Keep the same kernel geometry as catalogue batches while making each
    // timed sample long enough to stay below the protocol noise threshold.
    constexpr std::size_t result_count = 16'384U;
    constexpr unsigned int threads_per_block = 128U;
    constexpr std::uint32_t payment_count = 20U;
    const cir::ModelParameters model{{0.60f, 0.04f, 0.15f}, 0.03f};
    const std::vector<cir::ModelParameters> models(result_count, model);
    const std::vector<product::RegularEuropeanSwaptionParameters> products(
        result_count, {1.0f, 0.0325f, 0.5f, 252U, 126U, payment_count}
    );
    DeviceBuffer device_models(models.size() * sizeof(models.front()));
    DeviceBuffer device_products(products.size() * sizeof(products.front()));
    DeviceBuffer device_prices(result_count * sizeof(float));
    performance::copy_to_device(device_models, models);
    performance::copy_to_device(device_products, products);
    const auto launch = [&] {
        cir::launch_cir_european_swaption_cuda<SwaptionSide::payer>(
            device_models.as<cir::ModelParameters>(),
            result_count,
            device_products.as<product::RegularEuropeanSwaptionParameters>(),
            result_count,
            PriceConstruction::Aligned,
            result_count,
            0U,
            result_count,
            1.0f / 252.0f,
            threads_per_block,
            result_count,
            device_prices.as<float>(),
            payment_count
        );
    };
    const performance::Measurement measurement = performance::measure_cuda(
        launch, 5, 21
    );
    const std::vector<float> prices = performance::copy_from_device<float>(
        device_prices, result_count
    );
    bool finite = true;
    double price_sum = 0.0;
    for (const float price : prices) {
        finite = finite && std::isfinite(price) && price >= 0.0f;
        price_sum += price;
    }
    performance::emit_measurement(
        "PERF-004",
        "cir_noncentral_chi_square_inlining",
        AI_FACTORY_CIR_BENCHMARK_VARIANT,
        measurement,
        {{"result_count", result_count},
         {"payment_count", payment_count},
         {"threads_per_block", threads_per_block}},
        {{"finite_nonnegative", finite}, {"price_sum", price_sum}},
        5,
        21
    );
    return 0;
}
