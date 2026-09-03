// Representative one-/multi-factor Longstaff-Schwartz pipeline benchmarks.
#include "tests/performance/benchmark_support.cuh"

#include "common/fixed_income/swaption_side.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/option_side.cuh"
#include "model/equity/markovian/black_scholes/product/american_option.cuh"
#include "model/equity/markovian/heston/product/american_option.cuh"
#include "model/fixed_income/g2/product/bermudan_swaption.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/product/bermudan_swaption.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

using ai_factory::workbench::OptionSide;
using ai_factory::workbench::PriceConstruction;
using ai_factory::workbench::SwaptionSide;
using ai_factory::workbench::longstaff_schwartz::LaunchResult;
using ai_factory::workbench::performance::DeviceBuffer;
using ai_factory::workbench::performance::DeviceMemoryRole;
using ai_factory::workbench::performance::Measurement;
using ai_factory::workbench::performance::copy_from_device;
using ai_factory::workbench::performance::copy_to_device;
using ai_factory::workbench::performance::emit_measurement;
using ai_factory::workbench::performance::measure_synchronous_cuda_pipeline;

constexpr std::size_t kRowCount = 64U;
constexpr std::size_t kPathsPerPrice = 16'384U;
constexpr unsigned int kThreadsPerBlock = 128U;
constexpr std::size_t kBlocksPerPrice = 32U;
constexpr int kWarmups = 5;
constexpr int kRepetitions = 21;
constexpr std::size_t kDefaultOperationsPerTimingSample = 64U;
constexpr std::size_t kBlackScholesOperationsPerTimingSample = 128U;
constexpr std::size_t kOrnsteinUhlenbeckOperationsPerTimingSample = 256U;
constexpr float kDayFraction = 1.0f / 252.0f;
constexpr std::uint64_t kSeed = 2'170'000'001ULL;

template<typename ModelParameters, typename ProductParameters, typename Launch>
void benchmark_variant(
    const std::string& variant,
    const std::vector<ModelParameters>& models,
    const std::vector<ProductParameters>& products,
    nlohmann::ordered_json configuration,
    std::size_t operations_per_timing_sample,
    Launch&& launch
) {
    DeviceBuffer device_models(
        models.size() * sizeof(models.front()),
        DeviceMemoryRole::persistent_input
    );
    DeviceBuffer device_products(
        products.size() * sizeof(products.front()),
        DeviceMemoryRole::persistent_input
    );
    DeviceBuffer device_prices(
        kRowCount * sizeof(float), DeviceMemoryRole::output
    );
    DeviceBuffer device_errors(
        kRowCount * sizeof(float), DeviceMemoryRole::output
    );
    copy_to_device(device_models, models);
    copy_to_device(device_products, products);

    LaunchResult latest{};
    const auto timed_launch = [&] {
        latest = launch(
            device_models.template as<ModelParameters>(),
            products.data(),
            device_products.template as<ProductParameters>(),
            device_prices.as<float>(),
            device_errors.as<float>()
        );
        return latest.kernel_seconds * 1'000.0;
    };
    const Measurement measurement = measure_synchronous_cuda_pipeline(
        timed_launch, kWarmups, kRepetitions, operations_per_timing_sample
    );
    ai_factory::workbench::longstaff_schwartz::
        validate_regression_diagnostics(latest, variant.c_str());

    const std::vector<float> prices = copy_from_device<float>(
        device_prices, kRowCount
    );
    const std::vector<float> errors = copy_from_device<float>(
        device_errors, kRowCount
    );
    bool finite_nonnegative = true;
    double price_sum = 0.0;
    double error_sum = 0.0;
    for (std::size_t row = 0U; row < kRowCount; ++row) {
        finite_nonnegative = finite_nonnegative
            && std::isfinite(prices[row]) && prices[row] >= 0.0f
            && std::isfinite(errors[row]) && errors[row] >= 0.0f;
        price_sum += prices[row];
        error_sum += errors[row];
    }

    configuration["row_count"] = kRowCount;
    configuration["paths_per_price"] = kPathsPerPrice;
    configuration["threads_per_block"] = kThreadsPerBlock;
    configuration["blocks_per_price"] = kBlocksPerPrice;
    emit_measurement(
        "PERF-010",
        "longstaff_schwartz_pipeline",
        variant,
        measurement,
        std::move(configuration),
        {
            {"finite_nonnegative", finite_nonnegative},
            {"price_sum", price_sum},
            {"standard_error_sum", error_sum},
            {"kernel_launch_count", latest.kernel_launch_count},
            {"workspace_bytes", latest.workspace_bytes},
            {"successful_regression_count",
                latest.regression_diagnostics.successful_regression_count},
        },
        kWarmups,
        kRepetitions,
        {},
        latest.workspace_bytes
    );
}

void benchmark_equity_one_factor() {
    namespace model =
        ai_factory::workbench::model::equity::black_scholes;
    namespace product = ai_factory::workbench::product;
    const std::vector<model::ModelParameters> models(
        kRowCount, {1.0f, 0.03f, 0.01f, 0.25f}
    );
    const std::vector<product::AmericanOptionParameters> products(
        kRowCount, {1.05f, 252U, 21U}
    );
    benchmark_variant(
        "equity_one_factor_black_scholes",
        models,
        products,
        {{"exercise_dates", 12U}, {"transition", "exact"}},
        kBlackScholesOperationsPerTimingSample,
        [](const model::ModelParameters* device_models,
           const product::AmericanOptionParameters* host_products,
           const product::AmericanOptionParameters* device_products,
           float* prices,
           float* errors) {
            return model::launch_black_scholes_american_option_cuda<
                OptionSide::put
            >(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                PriceConstruction::Aligned, kRowCount, kPathsPerPrice,
                kDayFraction, kThreadsPerBlock, kBlocksPerPrice, kSeed,
                prices, errors
            );
        }
    );
}

void benchmark_equity_multi_state() {
    namespace model = ai_factory::workbench::model::equity::heston;
    namespace product = ai_factory::workbench::product;
    const std::vector<model::ModelParameters> models(
        kRowCount,
        {1.0f, 0.03f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f}
    );
    const std::vector<product::AmericanOptionParameters> products(
        kRowCount, {1.05f, 252U, 21U}
    );
    benchmark_variant(
        "equity_multi_state_heston",
        models,
        products,
        {{"exercise_dates", 12U}, {"transition", "fixed_step_qem"}},
        kDefaultOperationsPerTimingSample,
        [](const model::ModelParameters* device_models,
           const product::AmericanOptionParameters* host_products,
           const product::AmericanOptionParameters* device_products,
           float* prices,
           float* errors) {
            return model::launch_heston_american_option_cuda<OptionSide::put>(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                PriceConstruction::Aligned, kRowCount, kPathsPerPrice,
                kDayFraction, 1U,
                kThreadsPerBlock, kBlocksPerPrice, kSeed,
                prices, errors
            );
        }
    );
}

void benchmark_fixed_income_one_factor() {
    namespace model = ai_factory::workbench::model::fixed_income::
        ornstein_uhlenbeck;
    namespace product = ai_factory::workbench::product;
    const std::vector<model::ModelParameters> models(
        kRowCount, {{0.35f, 0.02f}, 0.03f}
    );
    const std::vector<product::BermudanSwaptionParameters> products(
        kRowCount, {1.0f, 0.03f, 0.5f, 126U, 126U, 8U, 8U}
    );
    benchmark_variant(
        "fixed_income_one_factor_ou",
        models,
        products,
        {{"exercise_dates", 8U}, {"rate_factors", 1U}},
        kOrnsteinUhlenbeckOperationsPerTimingSample,
        [](const model::ModelParameters* device_models,
           const product::BermudanSwaptionParameters* host_products,
           const product::BermudanSwaptionParameters* device_products,
           float* prices,
           float* errors) {
            return model::launch_ornstein_uhlenbeck_bermudan_swaption_cuda<
                SwaptionSide::payer
            >(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                PriceConstruction::Aligned, kRowCount, kPathsPerPrice,
                kDayFraction, kThreadsPerBlock, kBlocksPerPrice, kSeed,
                prices, errors
            );
        }
    );
}

void benchmark_fixed_income_two_factor() {
    namespace model = ai_factory::workbench::model::fixed_income::g2;
    namespace product = ai_factory::workbench::product;
    const std::vector<model::ModelParameters> models(
        kRowCount,
        {{0.20f, 0.005f, 0.75f, 0.012f, -0.40f}, {0.02f, 0.01f}}
    );
    const std::vector<product::BermudanSwaptionParameters> products(
        kRowCount, {1.0f, 0.03f, 0.5f, 126U, 126U, 8U, 8U}
    );
    benchmark_variant(
        "fixed_income_two_factor_g2",
        models,
        products,
        {{"exercise_dates", 8U}, {"rate_factors", 2U}},
        kDefaultOperationsPerTimingSample,
        [](const model::ModelParameters* device_models,
           const product::BermudanSwaptionParameters* host_products,
           const product::BermudanSwaptionParameters* device_products,
           float* prices,
           float* errors) {
            return model::launch_g2_bermudan_swaption_cuda<
                SwaptionSide::payer
            >(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                PriceConstruction::Aligned, kRowCount, kPathsPerPrice,
                kDayFraction, kThreadsPerBlock, kBlocksPerPrice, kSeed,
                prices, errors
            );
        }
    );
}

}  // namespace

int main() {
    benchmark_equity_one_factor();
    benchmark_equity_multi_state();
    benchmark_fixed_income_one_factor();
    benchmark_fixed_income_two_factor();
    return 0;
}
