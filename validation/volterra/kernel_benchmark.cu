// Kernel-only timing harness for Heston and rough-volatility call pricing.
#include "common/check_cuda.cuh"
#include "validation/performance/benchmark_support.cuh"
#include "model/equity/markovian/heston/european_option.cuh"
#include "model/equity/rough/rough_bergomi/european_option.cuh"
#include "model/equity/rough/rough_heston/european_option.cuh"
#include "model/equity/rough/rough_heston/numerics.hpp"
#include "model/equity/rough/rough_sabr/european_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ai_factory::workbench::OptionSide;
using ai_factory::workbench::PriceConstruction;
using ai_factory::workbench::check_cuda;
using ai_factory::workbench::product::EuropeanOptionParameters;

constexpr float day_fraction = 1.0f / 252.0f;
constexpr std::size_t rough_heston_factor_count = 7U;
constexpr std::uint64_t seed = 932000001ULL;
#ifndef AI_FACTORY_VOLTERRA_BENCHMARK_VARIANT
#define AI_FACTORY_VOLTERRA_BENCHMARK_VARIANT "hybrid_fft"
#endif

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes), "benchmark cudaMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) cudaFree(pointer_);
    }

    template<typename Value>
    Value* as() {
        return static_cast<Value*>(pointer_);
    }

private:
    void* pointer_ = nullptr;
};

template<typename Value>
void copy_to_device(DeviceBuffer& destination, const std::vector<Value>& source) {
    check_cuda(
        cudaMemcpy(
            destination.as<Value>(),
            source.data(),
            source.size() * sizeof(Value),
            cudaMemcpyHostToDevice
        ),
        "benchmark cudaMemcpy host to device"
    );
}

struct TimingResult {
    ai_factory::workbench::performance::Measurement measurement;
    float price;
    float standard_error;
};

template<typename Launch>
TimingResult measure(
    Launch&& launch,
    int repetitions,
    std::size_t price_count,
    float* device_prices,
    float* device_standard_errors
) {
    const auto measurement =
        ai_factory::workbench::performance::measure_cuda(
            launch, 5, repetitions
        );

    float price = 0.0f;
    float standard_error = 0.0f;
    check_cuda(
        cudaMemcpy(
            &price,
            device_prices,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "benchmark copy price"
    );
    check_cuda(
        cudaMemcpy(
            &standard_error,
            device_standard_errors,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "benchmark copy standard error"
    );
    if (!std::isfinite(price) || !std::isfinite(standard_error)
        || price < 0.0f || standard_error <= 0.0f) {
        throw std::runtime_error("benchmark produced invalid price moments");
    }
    return {
        measurement,
        price,
        standard_error,
    };
}

std::vector<EuropeanOptionParameters> products(
    std::size_t price_count,
    std::uint32_t maturity_days
) {
    return std::vector<EuropeanOptionParameters>(
        price_count, EuropeanOptionParameters{1.0f, maturity_days}
    );
}

void print_result(
    const char* model,
    std::uint32_t maturity_days,
    std::uint32_t steps_per_day,
    std::size_t step_count,
    std::size_t path_count,
    std::size_t price_count,
    std::size_t tuning_value,
    const char* tuning_name,
    int repetitions,
    const TimingResult& result
) {
    nlohmann::ordered_json configuration{
        {"model", model},
        {"maturity_days", maturity_days},
        {"steps_per_day", steps_per_day},
        {"time_steps", step_count},
        {"paths_per_price", path_count},
        {"price_count", price_count},
        {tuning_name, tuning_value},
    };
    ai_factory::workbench::performance::emit_measurement(
        "PERF-012/PERF-013/PERF-014",
        "volterra_pricing_pipeline",
        model,
        result.measurement,
        std::move(configuration),
        {{"price", result.price},
         {"standard_error", result.standard_error},
         {"median_ms_per_price",
            result.measurement.kernel.median_ms
                / static_cast<double>(price_count)},
         {"finite", std::isfinite(result.price)
            && std::isfinite(result.standard_error)}},
        5,
        repetitions
    );
}

void benchmark_heston(
    std::uint32_t maturity_days,
    std::uint32_t steps_per_day,
    std::size_t price_count,
    unsigned int threads,
    int repetitions,
    std::size_t path_count
) {
    using namespace ai_factory::workbench::model::equity::heston;
    const float dt = day_fraction / static_cast<float>(steps_per_day);
    const ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 0.30f,
        0.02f / 0.30f, 0.30f, -0.70f,
    };
    const std::vector<ModelParameters> models(price_count, model);
    const auto host_products = products(price_count, maturity_days);
    DeviceBuffer device_models(models.size() * sizeof(ModelParameters));
    DeviceBuffer device_products(
        host_products.size() * sizeof(EuropeanOptionParameters)
    );
    DeviceBuffer device_prices(price_count * sizeof(float));
    DeviceBuffer device_errors(price_count * sizeof(float));
    copy_to_device(device_models, models);
    copy_to_device(device_products, host_products);
    auto launch = [&] {
        launch_heston_european_option_cuda<OptionSide::call>(
            device_models.as<ModelParameters>(),
            price_count,
            device_products.as<EuropeanOptionParameters>(),
            price_count,
            PriceConstruction::Aligned,
            price_count,
            0U,
            price_count,
            path_count,
            dt,
            steps_per_day,
            threads,
            price_count,
            seed,
            device_prices.as<float>(),
            device_errors.as<float>()
        );
    };
    const TimingResult timing = measure(
        launch,
        repetitions,
        price_count,
        device_prices.as<float>(),
        device_errors.as<float>()
    );
    print_result(
        "heston",
        maturity_days,
        steps_per_day,
        static_cast<std::size_t>(maturity_days) * steps_per_day,
        path_count,
        price_count,
        threads,
        "threads",
        repetitions,
        timing
    );
}

void benchmark_rough_heston(
    std::uint32_t maturity_days,
    std::uint32_t steps_per_day,
    std::size_t price_count,
    unsigned int threads,
    int repetitions,
    std::size_t path_count
) {
    namespace rough =
        ai_factory::workbench::model::equity::rough_heston;
    const float dt = day_fraction / static_cast<float>(steps_per_day);
    const float horizon = static_cast<float>(maturity_days) * day_fraction;
    const rough::ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 0.30f,
        0.02f, 0.30f, 0.10f, -0.70f,
    };
    const std::vector<rough::ModelParameters> models(price_count, model);
    const auto prepared = rough::prepare_dynamics<rough_heston_factor_count>(
        models, horizon, dt
    );
    const auto host_products = products(price_count, maturity_days);
    DeviceBuffer device_models(models.size() * sizeof(rough::ModelParameters));
    DeviceBuffer device_prepared(
        prepared.size()
            * sizeof(rough::PreparedDynamics<rough_heston_factor_count>)
    );
    DeviceBuffer device_products(
        host_products.size() * sizeof(EuropeanOptionParameters)
    );
    DeviceBuffer device_prices(price_count * sizeof(float));
    DeviceBuffer device_errors(price_count * sizeof(float));
    copy_to_device(device_models, models);
    copy_to_device(device_prepared, prepared);
    copy_to_device(device_products, host_products);
    auto launch = [&] {
        rough::launch_rough_heston_european_option_cuda<
            OptionSide::call, rough_heston_factor_count
        >(
            device_models.as<rough::ModelParameters>(),
            price_count,
            device_prepared.as<
                rough::PreparedDynamics<rough_heston_factor_count>
            >(),
            price_count,
            device_products.as<EuropeanOptionParameters>(),
            price_count,
            PriceConstruction::Aligned,
            price_count,
            0U,
            price_count,
            path_count,
            dt,
            steps_per_day,
            threads,
            price_count,
            seed,
            device_prices.as<float>(),
            device_errors.as<float>()
        );
    };
    const TimingResult timing = measure(
        launch,
        repetitions,
        price_count,
        device_prices.as<float>(),
        device_errors.as<float>()
    );
    print_result(
        "rough_heston_n7",
        maturity_days,
        steps_per_day,
        static_cast<std::size_t>(maturity_days) * steps_per_day,
        path_count,
        price_count,
        threads,
        "threads",
        repetitions,
        timing
    );
}

void benchmark_rough_bergomi(
    std::uint32_t maturity_days,
    std::uint32_t steps_per_day,
    std::size_t price_count,
    std::size_t path_chunk_size,
    int repetitions,
    std::size_t path_count
) {
    namespace rough =
        ai_factory::workbench::model::equity::rough_bergomi;
    const float target_dt = day_fraction / static_cast<float>(steps_per_day);
    const std::size_t step_count =
        static_cast<std::size_t>(maturity_days) * steps_per_day;
    const rough::ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 1.70f, 0.10f, -0.70f,
    };
    const std::vector<rough::ModelParameters> models(
        price_count, model
    );
    const auto host_products = products(price_count, maturity_days);
    const rough::WorkspacePlan plan = rough::plan_pricing_workspace(
        step_count, path_count, path_chunk_size
    );
    DeviceBuffer device_models(
        models.size() * sizeof(rough::ModelParameters)
    );
    DeviceBuffer device_products(
        host_products.size() * sizeof(EuropeanOptionParameters)
    );
    DeviceBuffer device_workspace(plan.workspace_bytes);
    DeviceBuffer device_prices(price_count * sizeof(float));
    DeviceBuffer device_errors(price_count * sizeof(float));
    copy_to_device(device_models, models);
    copy_to_device(device_products, host_products);
    auto launch = [&] {
        for (std::size_t result_index = 0U;
             result_index < price_count;
             ++result_index) {
            rough::launch_rough_bergomi_european_option_cuda<OptionSide::call>(
                device_models.as<rough::ModelParameters>(),
                price_count,
                device_products.as<EuropeanOptionParameters>(),
                price_count,
                PriceConstruction::Aligned,
                price_count,
                result_index,
                path_count,
                day_fraction,
                target_dt,
                step_count,
                path_chunk_size,
                device_workspace.as<void>(),
                plan.workspace_bytes,
                seed,
                device_prices.as<float>(),
                device_errors.as<float>()
            );
        }
    };
    const TimingResult timing = measure(
        launch,
        repetitions,
        price_count,
        device_prices.as<float>(),
        device_errors.as<float>()
    );
    print_result(
        "rough_bergomi_" AI_FACTORY_VOLTERRA_BENCHMARK_VARIANT,
        maturity_days,
        steps_per_day,
        step_count,
        path_count,
        price_count,
        path_chunk_size,
        "path_chunk_size",
        repetitions,
        timing
    );
}

void benchmark_rough_sabr(
    std::uint32_t maturity_days,
    std::uint32_t steps_per_day,
    std::size_t price_count,
    std::size_t path_chunk_size,
    int repetitions,
    std::size_t path_count
) {
    namespace rough = ai_factory::workbench::model::equity::rough_sabr;
    const float target_dt = day_fraction / static_cast<float>(steps_per_day);
    const std::size_t step_count =
        static_cast<std::size_t>(maturity_days) * steps_per_day;
    const rough::ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 1.70f, 0.10f, -0.70f, 0.70f,
    };
    const std::vector<rough::ModelParameters> models(price_count, model);
    const auto host_products = products(price_count, maturity_days);
    const rough::WorkspacePlan plan = rough::plan_pricing_workspace(
        step_count, path_count, path_chunk_size
    );
    DeviceBuffer device_models(models.size() * sizeof(rough::ModelParameters));
    DeviceBuffer device_products(
        host_products.size() * sizeof(EuropeanOptionParameters)
    );
    DeviceBuffer device_workspace(plan.workspace_bytes);
    DeviceBuffer device_prices(price_count * sizeof(float));
    DeviceBuffer device_errors(price_count * sizeof(float));
    copy_to_device(device_models, models);
    copy_to_device(device_products, host_products);
    auto launch = [&] {
        for (std::size_t result_index = 0U;
             result_index < price_count;
             ++result_index) {
            rough::launch_rough_sabr_european_option_cuda<OptionSide::call>(
                device_models.as<rough::ModelParameters>(),
                price_count,
                device_products.as<EuropeanOptionParameters>(),
                price_count,
                PriceConstruction::Aligned,
                price_count,
                result_index,
                path_count,
                day_fraction,
                target_dt,
                step_count,
                path_chunk_size,
                device_workspace.as<void>(),
                plan.workspace_bytes,
                seed,
                device_prices.as<float>(),
                device_errors.as<float>()
            );
        }
    };
    const TimingResult timing = measure(
        launch,
        repetitions,
        price_count,
        device_prices.as<float>(),
        device_errors.as<float>()
    );
    print_result(
        "rough_sabr_" AI_FACTORY_VOLTERRA_BENCHMARK_VARIANT,
        maturity_days,
        steps_per_day,
        step_count,
        path_count,
        price_count,
        path_chunk_size,
        "path_chunk_size",
        repetitions,
        timing
    );
}

std::size_t parse_size(const char* value, const char* name) {
    try {
        return std::stoull(value);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
}

}  // namespace

int main(int argument_count, char** arguments) {
    if (argument_count < 7 || argument_count > 8) {
        std::fprintf(
            stderr,
            "usage: kernel_benchmark MODEL MATURITY_DAYS STEPS_PER_DAY "
            "PRICE_COUNT TUNING REPETITIONS [PATH_COUNT]\n"
        );
        return 2;
    }
    const std::string model = arguments[1];
    const auto maturity_days = static_cast<std::uint32_t>(
        parse_size(arguments[2], "maturity")
    );
    const auto steps_per_day = static_cast<std::uint32_t>(
        parse_size(arguments[3], "steps per day")
    );
    const std::size_t price_count = parse_size(arguments[4], "price count");
    const std::size_t tuning = parse_size(arguments[5], "tuning value");
    const int repetitions = static_cast<int>(
        parse_size(arguments[6], "repetitions")
    );
    const std::size_t path_count = argument_count == 8
        ? parse_size(arguments[7], "path count")
        : 1U << 20U;
    if (maturity_days == 0U || steps_per_day == 0U || price_count == 0U
        || tuning == 0U || repetitions < 3 || path_count < 2U) {
        throw std::invalid_argument("benchmark arguments must be positive");
    }

    if (model == "heston") {
        benchmark_heston(
            maturity_days,
            steps_per_day,
            price_count,
            static_cast<unsigned int>(tuning),
            repetitions,
            path_count
        );
    } else if (model == "rough_heston") {
        benchmark_rough_heston(
            maturity_days,
            steps_per_day,
            price_count,
            static_cast<unsigned int>(tuning),
            repetitions,
            path_count
        );
    } else if (model == "rough_bergomi") {
        benchmark_rough_bergomi(
            maturity_days,
            steps_per_day,
            price_count,
            tuning,
            repetitions,
            path_count
        );
    } else if (model == "rough_sabr") {
        benchmark_rough_sabr(
            maturity_days,
            steps_per_day,
            price_count,
            tuning,
            repetitions,
            path_count
        );
    } else {
        throw std::invalid_argument(
            "model must be heston, rough_heston, rough_bergomi, or rough_sabr"
        );
    }
    return 0;
}
