// Reproducible CUDA benchmark statistics and versioned NDJSON reporting.
#pragma once

#include "common/check_cuda.cuh"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ai_factory::workbench::performance {

inline constexpr int kProtocolVersion = 1;
inline constexpr int kDefaultWarmups = 5;
inline constexpr int kDefaultRepetitions = 21;
inline constexpr double kMinimumAcceptedGain = 0.05;
inline constexpr double kMaximumNoiseCoefficient = 0.05;

struct TimingStatistics {
    double minimum_ms;
    double median_ms;
    double p95_ms;
    double mean_ms;
    double standard_deviation_ms;
    double coefficient_of_variation;
};

struct Measurement {
    TimingStatistics kernel;
    TimingStatistics wall;
    TimingStatistics raw_host_clock;
};

inline TimingStatistics summarize(std::vector<double> samples) {
    if (samples.empty()) {
        throw std::invalid_argument("A benchmark requires timing samples.");
    }
    std::sort(samples.begin(), samples.end());
    const double mean = std::accumulate(
        samples.begin(), samples.end(), 0.0
    ) / static_cast<double>(samples.size());
    double squared_deviation = 0.0;
    for (const double sample : samples) {
        const double deviation = sample - mean;
        squared_deviation += deviation * deviation;
    }
    const double standard_deviation = std::sqrt(
        squared_deviation / static_cast<double>(samples.size())
    );
    const std::size_t p95_index = static_cast<std::size_t>(std::ceil(
        0.95 * static_cast<double>(samples.size())
    )) - 1U;
    return {
        samples.front(),
        samples[samples.size() / 2U],
        samples[p95_index],
        mean,
        standard_deviation,
        mean == 0.0 ? 0.0 : standard_deviation / mean,
    };
}

template<typename Launch>
Measurement measure_cuda(
    Launch&& launch,
    int warmups = kDefaultWarmups,
    int repetitions = kDefaultRepetitions
) {
    if (warmups < 1 || repetitions < 3) {
        throw std::invalid_argument(
            "A benchmark requires at least one warmup and three repetitions."
        );
    }
    for (int warmup = 0; warmup < warmups; ++warmup) launch();
    check_cuda(cudaDeviceSynchronize(), "performance warmup synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check_cuda(cudaEventCreate(&start), "performance create start event");
    check_cuda(cudaEventCreate(&stop), "performance create stop event");
    std::vector<double> kernel_samples;
    std::vector<double> wall_samples;
    std::vector<double> raw_host_samples;
    kernel_samples.reserve(static_cast<std::size_t>(repetitions));
    wall_samples.reserve(static_cast<std::size_t>(repetitions));
    raw_host_samples.reserve(static_cast<std::size_t>(repetitions));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto wall_start = std::chrono::steady_clock::now();
        check_cuda(cudaEventRecord(start), "performance record start");
        launch();
        check_cuda(cudaEventRecord(stop), "performance record stop");
        check_cuda(
            cudaEventSynchronize(stop), "performance synchronize stop"
        );
        check_cuda(
            cudaDeviceSynchronize(), "performance enclosing wall synchronize"
        );
        float kernel_milliseconds = 0.0f;
        check_cuda(
            cudaEventElapsedTime(&kernel_milliseconds, start, stop),
            "performance elapsed time"
        );
        const auto wall_stop = std::chrono::steady_clock::now();
        const double wall_milliseconds = std::chrono::duration<
            double, std::milli
        >(wall_stop - wall_start).count();
        kernel_samples.push_back(kernel_milliseconds);
        raw_host_samples.push_back(wall_milliseconds);
        // CUDA events and the CPU steady clock can differ under dynamic GPU
        // clocks. Preserve the raw clock sample, while the semantic enclosing
        // wall interval uses the device interval as its strict lower bound.
        wall_samples.push_back(std::max(
            wall_milliseconds,
            static_cast<double>(kernel_milliseconds)
        ));
    }
    check_cuda(cudaEventDestroy(stop), "performance destroy stop event");
    check_cuda(cudaEventDestroy(start), "performance destroy start event");
    return {
        summarize(kernel_samples),
        summarize(wall_samples),
        summarize(raw_host_samples),
    };
}

inline nlohmann::ordered_json timing_json(
    const TimingStatistics& statistics
) {
    return {
        {"minimum_ms", statistics.minimum_ms},
        {"median_ms", statistics.median_ms},
        {"p95_ms", statistics.p95_ms},
        {"mean_ms", statistics.mean_ms},
        {"standard_deviation_ms", statistics.standard_deviation_ms},
        {"coefficient_of_variation", statistics.coefficient_of_variation},
    };
}

inline nlohmann::ordered_json environment_json() {
    int device_index = 0;
    check_cuda(cudaGetDevice(&device_index), "performance get device");
    cudaDeviceProp properties{};
    check_cuda(
        cudaGetDeviceProperties(&properties, device_index),
        "performance get device properties"
    );
    int driver_version = 0;
    int runtime_version = 0;
    check_cuda(
        cudaDriverGetVersion(&driver_version), "performance driver version"
    );
    check_cuda(
        cudaRuntimeGetVersion(&runtime_version), "performance runtime version"
    );
    return {
        {"gpu", properties.name},
        {"compute_capability",
            std::to_string(properties.major) + "."
                + std::to_string(properties.minor)},
        {"sm_count", properties.multiProcessorCount},
        {"memory_bytes", properties.totalGlobalMem},
        {"pci_bus_id", properties.pciBusID},
        {"driver_version", driver_version},
        {"runtime_version", runtime_version},
        {"cuda_compiler_version",
            std::to_string(__CUDACC_VER_MAJOR__) + "."
                + std::to_string(__CUDACC_VER_MINOR__) + "."
                + std::to_string(__CUDACC_VER_BUILD__)},
    };
}

inline void emit_measurement(
    const std::string& finding,
    const std::string& benchmark,
    const std::string& variant,
    const Measurement& measurement,
    nlohmann::ordered_json configuration,
    nlohmann::ordered_json numerical_check = nlohmann::ordered_json::object(),
    int warmups = kDefaultWarmups,
    int repetitions = kDefaultRepetitions
) {
    const nlohmann::ordered_json report{
        {"schema", "ai_factory_cuda_performance_baseline"},
        {"protocol_version", kProtocolVersion},
        {"finding", finding},
        {"benchmark", benchmark},
        {"variant", variant},
        {"environment", environment_json()},
        {"protocol", {
            {"warmups", warmups},
            {"repetitions", repetitions},
            {"primary_statistic", "median_ms"},
            {"tail_statistic", "p95_ms"},
            {"minimum_accepted_gain", kMinimumAcceptedGain},
            {"maximum_noise_coefficient", kMaximumNoiseCoefficient},
            {"wall_semantics",
                "max(raw_host_clock, enclosing_cuda_event_interval)"},
        }},
        {"configuration", std::move(configuration)},
        {"kernel", timing_json(measurement.kernel)},
        {"wall", timing_json(measurement.wall)},
        {"raw_host_clock", timing_json(measurement.raw_host_clock)},
        {"numerical_check", std::move(numerical_check)},
    };
    std::cout << report.dump() << '\n';
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes), "performance cudaMalloc");
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) static_cast<void>(cudaFree(pointer_));
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
        "performance copy to device"
    );
}

template<typename Value>
std::vector<Value> copy_from_device(
    DeviceBuffer& source,
    std::size_t count
) {
    std::vector<Value> values(count);
    check_cuda(
        cudaMemcpy(
            values.data(),
            source.as<Value>(),
            count * sizeof(Value),
            cudaMemcpyDeviceToHost
        ),
        "performance copy from device"
    );
    return values;
}

}  // namespace ai_factory::workbench::performance
