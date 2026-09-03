// Focused CUDA experiments for generic indexing, geometry, precision and overhead.
#include "validation/performance/benchmark_support.cuh"

#include "common/fixed_income/swaption_side.cuh"
#include "common/result_index.cuh"
#include "curve/nelson_siegel/parameters.hpp"
#include "model/equity/markovian/bates/phoenix_autocall.cuh"
#include "model/equity/markovian/bates/phoenix_memory_autocall.cuh"
#include "model/equity/markovian/black_scholes/european_option.cuh"
#include "model/fixed_income/hull_white/nelson_siegel/european_swaption.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ai_factory::workbench::PriceConstruction;
using ai_factory::workbench::performance::DeviceBuffer;
using ai_factory::workbench::performance::Measurement;
using ai_factory::workbench::performance::copy_from_device;
using ai_factory::workbench::performance::copy_to_device;
using ai_factory::workbench::performance::emit_measurement;
using ai_factory::workbench::performance::measure_cuda;

constexpr unsigned int kThreads = 256U;
constexpr int kWarmups = 5;
constexpr int kRepetitions = 21;

template<PriceConstruction Construction>
__global__ void specialized_index_kernel(
    std::size_t result_count,
    std::size_t product_count,
    std::uint32_t iterations,
    std::uint64_t* output
) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= result_count) return;
    std::uint64_t checksum = 0U;
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const std::size_t result = (
            thread + static_cast<std::size_t>(iteration) * 7'919U
        ) % result_count;
        const auto indices =
            ai_factory::workbench::decode_model_product_result_index(
                result, product_count, Construction
            );
        checksum += indices.model_index * 65'537U + indices.product_index;
    }
    output[thread] = checksum;
}

__global__ void runtime_index_kernel(
    std::size_t result_count,
    std::size_t product_count,
    PriceConstruction construction,
    std::uint32_t iterations,
    std::uint64_t* output
) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= result_count) return;
    std::uint64_t checksum = 0U;
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const std::size_t result = (
            thread + static_cast<std::size_t>(iteration) * 7'919U
        ) % result_count;
        const auto indices =
            ai_factory::workbench::decode_model_product_result_index(
                result, product_count, construction
            );
        checksum += indices.model_index * 65'537U + indices.product_index;
    }
    output[thread] = checksum;
}

__global__ void index_32_kernel(
    std::uint32_t result_count,
    std::uint32_t product_count,
    std::uint32_t iterations,
    std::uint64_t* output
) {
    const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= result_count) return;
    std::uint64_t checksum = 0U;
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const std::uint32_t result = (
            thread + iteration * 7'919U
        ) % result_count;
        checksum += static_cast<std::uint64_t>(result / product_count)
                * 65'537U
            + result % product_count;
    }
    output[thread] = checksum;
}

std::uint64_t checksum(const std::vector<std::uint64_t>& values) {
    std::uint64_t total = 0U;
    for (const std::uint64_t value : values) total ^= value;
    return total;
}

void benchmark_indexing() {
    constexpr std::size_t result_count = 1U << 20U;
    constexpr std::size_t product_count = 1'000U;
    constexpr std::uint32_t iterations = 1'024U;
    const std::size_t block_count =
        (result_count + kThreads - 1U) / kThreads;
    DeviceBuffer output(result_count * sizeof(std::uint64_t));

    const auto runtime_launch = [&] {
        runtime_index_kernel<<<block_count, kThreads>>>(
            result_count,
            product_count,
            PriceConstruction::CartesianProduct,
            iterations,
            output.as<std::uint64_t>()
        );
    };
    const Measurement runtime = measure_cuda(
        runtime_launch, kWarmups, kRepetitions
    );
    const std::vector<std::uint64_t> reference =
        copy_from_device<std::uint64_t>(output, result_count);
    emit_measurement(
        "PERF-001",
        "result_index_decode",
        "runtime_enum_size_t",
        runtime,
        {{"result_count", result_count}, {"product_count", product_count},
         {"iterations_per_thread", iterations}},
        {{"checksum", checksum(reference)}, {"mapping_identical", true}},
        kWarmups,
        kRepetitions
    );

    const auto specialized_launch = [&] {
        specialized_index_kernel<PriceConstruction::CartesianProduct>
            <<<block_count, kThreads>>>(
                result_count,
                product_count,
                iterations,
                output.as<std::uint64_t>()
            );
    };
    const Measurement specialized = measure_cuda(
        specialized_launch, kWarmups, kRepetitions
    );
    const std::vector<std::uint64_t> specialized_values =
        copy_from_device<std::uint64_t>(output, result_count);
    emit_measurement(
        "PERF-001",
        "result_index_decode",
        "compile_time_mode_size_t",
        specialized,
        {{"result_count", result_count}, {"product_count", product_count},
         {"iterations_per_thread", iterations}},
        {{"checksum", checksum(specialized_values)},
         {"mapping_identical", specialized_values == reference}},
        kWarmups,
        kRepetitions
    );

    const auto index_32_launch = [&] {
        index_32_kernel<<<block_count, kThreads>>>(
            static_cast<std::uint32_t>(result_count),
            static_cast<std::uint32_t>(product_count),
            iterations,
            output.as<std::uint64_t>()
        );
    };
    const Measurement index_32 = measure_cuda(
        index_32_launch, kWarmups, kRepetitions
    );
    const std::vector<std::uint64_t> index_32_values =
        copy_from_device<std::uint64_t>(output, result_count);
    emit_measurement(
        "PERF-001",
        "result_index_decode",
        "validated_uint32",
        index_32,
        {{"result_count", result_count}, {"product_count", product_count},
         {"iterations_per_thread", iterations}},
        {{"checksum", checksum(index_32_values)},
         {"mapping_identical", index_32_values == reference}},
        kWarmups,
        kRepetitions
    );
}

enum class AccumulationMode : std::uint8_t {
    Fp64,
    CompensatedFp32,
    ChunkedFp32Fp64,
};

template<AccumulationMode Mode>
__global__ void accumulation_kernel(
    std::size_t value_count,
    std::uint32_t iterations,
    double* output
) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (thread >= value_count) return;
    double sum_64 = 0.0;
    float sum_32 = 0.0f;
    float correction = 0.0f;
    float chunk = 0.0f;
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        // A non-negative, mixed-scale payoff-like stream.  The sparse large
        // observations expose the precision lost when small path payoffs are
        // accumulated in FP32, without manufacturing cancellation.
        const float value = (iteration % 257U) == 0U
            ? 100'000.0f + static_cast<float>(thread % 17U)
            : 1.0e-3f
                * (1.0f
                    + static_cast<float>((thread + iteration) % 31U)
                        * 1.0e-2f);
        if constexpr (Mode == AccumulationMode::Fp64) {
            sum_64 += static_cast<double>(value);
        } else if constexpr (Mode == AccumulationMode::CompensatedFp32) {
            const float adjusted = value - correction;
            const float updated = sum_32 + adjusted;
            correction = (updated - sum_32) - adjusted;
            sum_32 = updated;
        } else {
            chunk += value;
            if ((iteration & 31U) == 31U) {
                sum_64 += static_cast<double>(chunk);
                chunk = 0.0f;
            }
        }
    }
    if constexpr (Mode == AccumulationMode::CompensatedFp32) {
        output[thread] = static_cast<double>(sum_32);
    } else {
        output[thread] = sum_64 + static_cast<double>(chunk);
    }
}

double maximum_absolute_error(
    const std::vector<double>& values,
    const std::vector<double>& reference
) {
    double maximum = 0.0;
    for (std::size_t index = 0U; index < values.size(); ++index) {
        maximum = std::max(maximum, std::fabs(values[index] - reference[index]));
    }
    return maximum;
}

double maximum_relative_error(
    const std::vector<double>& values,
    const std::vector<double>& reference
) {
    double maximum = 0.0;
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const double scale = std::max(std::fabs(reference[index]), 1.0e-300);
        maximum = std::max(
            maximum,
            std::fabs(values[index] - reference[index]) / scale
        );
    }
    return maximum;
}

template<AccumulationMode Mode>
Measurement measure_accumulation(
    DeviceBuffer& output,
    std::size_t value_count,
    std::uint32_t iterations
) {
    const std::size_t block_count = (value_count + kThreads - 1U) / kThreads;
    return measure_cuda(
        [&] {
            accumulation_kernel<Mode><<<block_count, kThreads>>>(
                value_count, iterations, output.as<double>()
            );
        },
        kWarmups,
        kRepetitions
    );
}

void benchmark_accumulation() {
    constexpr std::size_t value_count = 1U << 22U;
    constexpr std::uint32_t iterations = 1'024U;
    DeviceBuffer output(value_count * sizeof(double));
    const Measurement fp64 = measure_accumulation<AccumulationMode::Fp64>(
        output, value_count, iterations
    );
    const std::vector<double> reference =
        copy_from_device<double>(output, value_count);
    emit_measurement(
        "PERF-008", "hot_loop_accumulation", "fp64", fp64,
        {{"value_count", value_count}, {"iterations", iterations},
         {"input_profile", "non_negative_mixed_scale"}},
        {{"maximum_absolute_error", 0.0},
         {"maximum_relative_error", 0.0}},
        kWarmups,
        kRepetitions
    );

    const Measurement compensated =
        measure_accumulation<AccumulationMode::CompensatedFp32>(
            output, value_count, iterations
        );
    const std::vector<double> compensated_values =
        copy_from_device<double>(output, value_count);
    emit_measurement(
        "PERF-008",
        "hot_loop_accumulation",
        "compensated_fp32",
        compensated,
        {{"value_count", value_count}, {"iterations", iterations},
         {"input_profile", "non_negative_mixed_scale"}},
        {{"maximum_absolute_error",
            maximum_absolute_error(compensated_values, reference)},
         {"maximum_relative_error",
            maximum_relative_error(compensated_values, reference)}},
        kWarmups,
        kRepetitions
    );

    const Measurement chunked =
        measure_accumulation<AccumulationMode::ChunkedFp32Fp64>(
            output, value_count, iterations
        );
    const std::vector<double> chunked_values =
        copy_from_device<double>(output, value_count);
    emit_measurement(
        "PERF-008",
        "hot_loop_accumulation",
        "chunked_fp32_fp64",
        chunked,
        {{"value_count", value_count}, {"iterations", iterations},
         {"chunk_size", 32U},
         {"input_profile", "non_negative_mixed_scale"}},
        {{"maximum_absolute_error",
            maximum_absolute_error(chunked_values, reference)},
         {"maximum_relative_error",
            maximum_relative_error(chunked_values, reference)}},
        kWarmups,
        kRepetitions
    );
}

void benchmark_closed_form_overhead(std::size_t result_count) {
    namespace black_scholes =
        ai_factory::workbench::model::equity::black_scholes;
    using ai_factory::workbench::OptionSide;
    using ai_factory::workbench::product::EuropeanOptionParameters;
    const std::vector<black_scholes::ModelParameters> models(
        result_count, {1.0f, 0.02f, 0.01f, 0.20f}
    );
    const std::vector<EuropeanOptionParameters> products(
        result_count, {1.0f, 252U}
    );
    DeviceBuffer device_models(models.size() * sizeof(models.front()));
    DeviceBuffer device_products(products.size() * sizeof(products.front()));
    DeviceBuffer device_prices(result_count * sizeof(float));
    copy_to_device(device_models, models);
    copy_to_device(device_products, products);
    constexpr unsigned int threads = 128U;
    const std::size_t blocks = std::min(
        result_count,
        (result_count + threads - 1U) / threads
    );
    constexpr std::size_t launches_per_sample = 256U;
    const auto launch = [&] {
        for (std::size_t repetition = 0U;
             repetition < launches_per_sample;
             ++repetition) {
            black_scholes::launch_black_scholes_european_option_cuda<
                OptionSide::call
            >(
                device_models.as<black_scholes::ModelParameters>(),
                result_count,
                device_products.as<EuropeanOptionParameters>(),
                result_count,
                PriceConstruction::Aligned,
                result_count,
                0U,
                result_count,
                1.0f / 252.0f,
                threads,
                blocks,
                device_prices.as<float>()
            );
        }
    };
    const Measurement measurement = measure_cuda(
        launch, kWarmups, kRepetitions
    );
    const std::vector<float> prices = copy_from_device<float>(
        device_prices, result_count
    );
    emit_measurement(
        "PERF-009",
        "closed_form_launch_overhead",
        "validated_public_launcher",
        measurement,
        {{"result_count", result_count}, {"threads_per_block", threads},
         {"block_count", blocks},
         {"launches_per_sample", launches_per_sample}},
        {{"first_price", prices.front()},
         {"finite", std::isfinite(prices.front())},
         {"median_wall_per_launch_ms",
            measurement.wall.median_ms
                / static_cast<double>(launches_per_sample)},
         {"median_kernel_per_launch_ms",
            measurement.kernel.median_ms
                / static_cast<double>(launches_per_sample)}},
        kWarmups,
        kRepetitions
    );
}

template<typename Product, typename Launch>
void benchmark_rich_policy(
    const std::string& policy,
    const Product& product,
    Launch&& launch
) {
    namespace bates = ai_factory::workbench::model::equity::bates;
    constexpr std::size_t result_count = 32U;
    constexpr std::size_t paths_per_price = 4'096U;
    const std::vector<bates::ModelParameters> models(
        result_count,
        {1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f,
         0.10f, -0.05f, 0.20f}
    );
    const std::vector<Product> products(result_count, product);
    DeviceBuffer device_models(models.size() * sizeof(models.front()));
    DeviceBuffer device_products(products.size() * sizeof(products.front()));
    DeviceBuffer device_prices(result_count * sizeof(float));
    DeviceBuffer device_errors(result_count * sizeof(float));
    copy_to_device(device_models, models);
    copy_to_device(device_products, products);
    std::vector<float> reference;
    for (const unsigned int threads : {128U, 256U, 512U}) {
        const auto configured_launch = [&] {
            launch(
                device_models.as<bates::ModelParameters>(),
                device_products.as<Product>(),
                threads,
                device_prices.as<float>(),
                device_errors.as<float>(),
                result_count,
                paths_per_price
            );
        };
        const Measurement measurement = measure_cuda(
            configured_launch, kWarmups, kRepetitions
        );
        const std::vector<float> prices = copy_from_device<float>(
            device_prices, result_count
        );
        if (reference.empty()) reference = prices;
        float maximum_difference = 0.0f;
        for (std::size_t index = 0U; index < prices.size(); ++index) {
            maximum_difference = std::max(
                maximum_difference,
                std::fabs(prices[index] - reference[index])
            );
        }
        emit_measurement(
            "PERF-002",
            "rich_policy_geometry",
            policy + "_threads_" + std::to_string(threads),
            measurement,
            {{"result_count", result_count},
             {"paths_per_price", paths_per_price},
             {"threads_per_block", threads},
             {"block_count", result_count}},
            {{"first_price", prices.front()},
             {"maximum_price_difference", maximum_difference}},
            kWarmups,
            kRepetitions
        );
    }
}

void benchmark_rich_policies() {
    namespace bates = ai_factory::workbench::model::equity::bates;
    using ai_factory::workbench::product::PhoenixAutocallParameters;
    using ai_factory::workbench::product::PhoenixMemoryAutocallParameters;
    const auto common_launch = [](
        auto launcher,
        const bates::ModelParameters* models,
        const auto* products,
        unsigned int threads,
        float* prices,
        float* errors,
        std::size_t result_count,
        std::size_t paths_per_price
    ) {
        launcher(
            models,
            result_count,
            products,
            result_count,
            PriceConstruction::Aligned,
            result_count,
            0U,
            result_count,
            paths_per_price,
            1.0f / 504.0f,
            2U,
            threads,
            result_count,
            900000001ULL,
            prices,
            errors
        );
    };
    benchmark_rich_policy(
        "bates_phoenix",
        PhoenixAutocallParameters{252U, 21U, 1.0f, 0.70f, 0.60f, 0.08f},
        [&](const auto* models, const auto* products, unsigned int threads,
            float* prices, float* errors, std::size_t count,
            std::size_t paths) {
            common_launch(
                bates::launch_bates_phoenix_autocall_cuda,
                models, products, threads, prices, errors, count, paths
            );
        }
    );
    benchmark_rich_policy(
        "bates_phoenix_memory",
        PhoenixMemoryAutocallParameters{
            252U, 21U, 1.0f, 0.70f, 0.60f, 0.08f
        },
        [&](const auto* models, const auto* products, unsigned int threads,
            float* prices, float* errors, std::size_t count,
            std::size_t paths) {
            common_launch(
                bates::launch_bates_phoenix_memory_autocall_cuda,
                models, products, threads, prices, errors, count, paths
            );
        }
    );
}

enum class ScheduleVariant : std::uint8_t {
    Regular,
    ExplicitHomogeneous,
    ExplicitHeterogeneous,
};

void benchmark_ragged_schedules(ScheduleVariant variant) {
    namespace hull_white =
        ai_factory::workbench::model::fixed_income::hull_white;
    namespace pricing =
        ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel;
    namespace curve = ai_factory::workbench::curve::nelson_siegel;
    namespace product = ai_factory::workbench::product;
    using ai_factory::workbench::SwaptionSide;
    constexpr std::size_t result_count = 65'536U;
    const std::vector<hull_white::ModelParameters> models(
        result_count, {0.10f, 0.01f}
    );
    const std::vector<curve::NelsonSiegelParameters> curves(
        result_count, {0.03f, -0.02f, 0.02f, 2.0f}
    );
    DeviceBuffer device_models(models.size() * sizeof(models.front()));
    DeviceBuffer device_curves(curves.size() * sizeof(curves.front()));
    DeviceBuffer device_prices(result_count * sizeof(float));
    copy_to_device(device_models, models);
    copy_to_device(device_curves, curves);
    constexpr unsigned int threads = 256U;
    const std::size_t block_count =
        (result_count + threads - 1U) / threads;

    Measurement measurement{};
    std::vector<float> prices;
    std::string variant_name;
    std::size_t schedule_size = 0U;
    if (variant == ScheduleVariant::Regular) {
        const std::vector<product::RegularEuropeanSwaptionParameters> products(
            result_count, {1.0f, 0.03f, 0.5f, 252U, 126U, 20U}
        );
        DeviceBuffer device_products(products.size() * sizeof(products.front()));
        copy_to_device(device_products, products);
        measurement = measure_cuda(
            [&] {
                pricing::launch_hull_white_nelson_siegel_european_swaption_cuda<
                    SwaptionSide::payer
                >(
                    device_models.as<hull_white::ModelParameters>(),
                    result_count,
                    device_curves.as<curve::NelsonSiegelParameters>(),
                    result_count,
                    device_products.as<
                        product::RegularEuropeanSwaptionParameters
                    >(),
                    result_count,
                    PriceConstruction::Aligned,
                    result_count,
                    0U,
                    result_count,
                    1.0f / 252.0f,
                    threads,
                    block_count,
                    device_prices.as<float>()
                );
            },
            kWarmups,
            kRepetitions
        );
        prices = copy_from_device<float>(device_prices, result_count);
        variant_name = "regular_20_payments";
    } else {
        std::vector<product::ExplicitEuropeanSwaptionParameters> products;
        const std::uint32_t maximum_payment_count =
            variant == ScheduleVariant::ExplicitHomogeneous ? 20U : 30U;
        std::vector<std::uint32_t> payment_times(
            result_count * maximum_payment_count
        );
        std::vector<float> accrual_fractions(
            result_count * maximum_payment_count
        );
        products.reserve(result_count);
        for (std::size_t row = 0U; row < result_count; ++row) {
            const std::uint32_t payment_count =
                variant == ScheduleVariant::ExplicitHomogeneous
                ? 20U : 2U + static_cast<std::uint32_t>(row % 29U);
            products.push_back({
                1.0f, 0.03f, 252U, payment_count, row
            });
            for (std::uint32_t payment = 0U;
                 payment < payment_count;
                 ++payment) {
                const std::size_t pool_index =
                    static_cast<std::size_t>(payment) * result_count + row;
                payment_times[pool_index] =
                    252U + (payment + 1U) * 126U;
                accrual_fractions[pool_index] = 0.5f;
            }
        }
        schedule_size = payment_times.size();
        DeviceBuffer device_products(products.size() * sizeof(products.front()));
        DeviceBuffer device_payment_times(
            payment_times.size() * sizeof(payment_times.front())
        );
        DeviceBuffer device_accrual_fractions(
            accrual_fractions.size() * sizeof(accrual_fractions.front())
        );
        copy_to_device(device_products, products);
        copy_to_device(device_payment_times, payment_times);
        copy_to_device(device_accrual_fractions, accrual_fractions);
        measurement = measure_cuda(
            [&] {
                pricing::launch_hull_white_nelson_siegel_european_swaption_cuda<
                    SwaptionSide::payer
                >(
                    device_models.as<hull_white::ModelParameters>(),
                    result_count,
                    device_curves.as<curve::NelsonSiegelParameters>(),
                    result_count,
                    device_products.as<
                        product::ExplicitEuropeanSwaptionParameters
                    >(),
                    device_payment_times.as<std::uint32_t>(),
                    device_accrual_fractions.as<float>(),
                    schedule_size,
                    result_count,
                    PriceConstruction::Aligned,
                    result_count,
                    0U,
                    result_count,
                    1.0f / 252.0f,
                    threads,
                    result_count,
                    device_prices.as<float>(),
                    maximum_payment_count
                );
            },
            kWarmups,
            kRepetitions
        );
        prices = copy_from_device<float>(device_prices, result_count);
        variant_name = variant == ScheduleVariant::ExplicitHomogeneous
            ? "explicit_homogeneous_20_payments"
            : "explicit_heterogeneous_2_to_30_payments";
    }
    bool finite = true;
    double price_sum = 0.0;
    for (const float price : prices) {
        finite = finite && std::isfinite(price) && price >= 0.0f;
        price_sum += price;
    }
    emit_measurement(
        "PERF-006",
        "explicit_schedule_layout",
        variant_name,
        measurement,
        {{"result_count", result_count},
         {"schedule_pool_elements", schedule_size},
         {"threads_per_block", threads}},
        {{"finite_nonnegative", finite}, {"price_sum", price_sum}},
        kWarmups,
        kRepetitions
    );
}

std::size_t parse_size(const char* value) {
    try {
        return std::stoull(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid positive integer argument");
    }
}

}  // namespace

int main(int argument_count, char** arguments) {
    if (argument_count < 2) {
        throw std::invalid_argument(
            "usage: generic_kernel_benchmark "
            "index|accumulation|overhead [result_count]|geometry|"
            "ragged regular|homogeneous|heterogeneous"
        );
    }
    const std::string mode = arguments[1];
    if (mode == "index") {
        benchmark_indexing();
    } else if (mode == "accumulation") {
        benchmark_accumulation();
    } else if (mode == "overhead") {
        benchmark_closed_form_overhead(
            argument_count >= 3 ? parse_size(arguments[2]) : 1U
        );
    } else if (mode == "geometry") {
        benchmark_rich_policies();
    } else if (mode == "ragged") {
        if (argument_count != 3) {
            throw std::invalid_argument("ragged requires a schedule variant");
        }
        const std::string variant = arguments[2];
        benchmark_ragged_schedules(
            variant == "regular" ? ScheduleVariant::Regular
            : variant == "homogeneous"
                ? ScheduleVariant::ExplicitHomogeneous
                : variant == "heterogeneous"
                    ? ScheduleVariant::ExplicitHeterogeneous
                    : throw std::invalid_argument("unknown schedule variant")
        );
    } else {
        throw std::invalid_argument("unknown generic benchmark mode");
    }
    return 0;
}
