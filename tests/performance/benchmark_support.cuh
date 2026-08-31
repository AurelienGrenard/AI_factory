// Reproducible CUDA benchmark statistics and versioned NDJSON reporting.
#pragma once

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

inline constexpr int kProtocolVersion = 3;
inline constexpr int kDefaultWarmups = 5;
inline constexpr int kDefaultRepetitions = 21;
inline constexpr double kMinimumAcceptedGain = 0.05;
inline constexpr double kMaximumNoiseCoefficient = 0.05;
inline constexpr double kMaximumPublicationNoiseCoefficient = 0.10;

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
    std::size_t operations_per_sample;
};

inline TimingStatistics per_operation(
    TimingStatistics statistics,
    std::size_t operations_per_sample
) {
    if (operations_per_sample == 0U) {
        throw std::invalid_argument(
            "A timing sample requires at least one operation."
        );
    }
    const double divisor = static_cast<double>(operations_per_sample);
    statistics.minimum_ms /= divisor;
    statistics.median_ms /= divisor;
    statistics.p95_ms /= divisor;
    statistics.mean_ms /= divisor;
    statistics.standard_deviation_ms /= divisor;
    return statistics;
}

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
    int repetitions = kDefaultRepetitions,
    std::size_t operations_per_sample = 1U
) {
    if (warmups < 1 || repetitions < 3 || operations_per_sample == 0U) {
        throw std::invalid_argument(
            "A benchmark requires warmups, repetitions, and timed operations."
        );
    }
    for (int warmup = 0; warmup < warmups; ++warmup) {
        for (std::size_t operation = 0U;
             operation < operations_per_sample;
             ++operation) {
            launch();
        }
    }
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
        for (std::size_t operation = 0U;
             operation < operations_per_sample;
             ++operation) {
            launch();
        }
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
        per_operation(summarize(kernel_samples), operations_per_sample),
        per_operation(summarize(wall_samples), operations_per_sample),
        per_operation(summarize(raw_host_samples), operations_per_sample),
        operations_per_sample,
    };
}

// Measure a host-orchestrated pipeline whose public launcher synchronizes and
// returns its own enclosing CUDA-event interval in milliseconds. This avoids
// counting host-side planning and event construction as kernel time while the
// wall statistic continues to enclose the complete public call.
template<typename Launch>
Measurement measure_synchronous_cuda_pipeline(
    Launch&& launch,
    int warmups = kDefaultWarmups,
    int repetitions = kDefaultRepetitions,
    std::size_t operations_per_sample = 1U
) {
    if (warmups < 1 || repetitions < 3 || operations_per_sample == 0U) {
        throw std::invalid_argument(
            "A pipeline benchmark requires warmups, repetitions, and operations."
        );
    }
    for (int warmup = 0; warmup < warmups; ++warmup) {
        for (std::size_t operation = 0U;
             operation < operations_per_sample;
             ++operation) {
            static_cast<void>(launch());
        }
    }
    check_cuda(cudaDeviceSynchronize(), "pipeline warmup synchronize");

    std::vector<double> kernel_samples;
    std::vector<double> wall_samples;
    std::vector<double> raw_host_samples;
    kernel_samples.reserve(static_cast<std::size_t>(repetitions));
    wall_samples.reserve(static_cast<std::size_t>(repetitions));
    raw_host_samples.reserve(static_cast<std::size_t>(repetitions));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto wall_start = std::chrono::steady_clock::now();
        double kernel_milliseconds = 0.0;
        for (std::size_t operation = 0U;
             operation < operations_per_sample;
             ++operation) {
            kernel_milliseconds += launch();
        }
        const auto wall_stop = std::chrono::steady_clock::now();
        const double wall_milliseconds = std::chrono::duration<
            double, std::milli
        >(wall_stop - wall_start).count();
        kernel_samples.push_back(kernel_milliseconds);
        raw_host_samples.push_back(wall_milliseconds);
        wall_samples.push_back(std::max(
            wall_milliseconds, kernel_milliseconds
        ));
    }
    return {
        per_operation(summarize(kernel_samples), operations_per_sample),
        per_operation(summarize(wall_samples), operations_per_sample),
        per_operation(summarize(raw_host_samples), operations_per_sample),
        operations_per_sample,
    };
}

template<typename Operation>
TimingStatistics measure_host(
    Operation&& operation,
    int warmups = kDefaultWarmups,
    int repetitions = kDefaultRepetitions
) {
    if (warmups < 1 || repetitions < 3) {
        throw std::invalid_argument(
            "A host benchmark requires at least one warmup and three repetitions."
        );
    }
    for (int warmup = 0; warmup < warmups; ++warmup) operation();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repetitions));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto start = std::chrono::steady_clock::now();
        operation();
        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(
            stop - start
        ).count());
    }
    return summarize(std::move(samples));
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

enum class DeviceMemoryRole : std::size_t {
    persistent_input = 0U,
    caller_workspace = 1U,
    output = 2U,
    count = 3U,
};

struct DeviceMemoryTracker {
    std::array<std::size_t, 3U> live{};
    std::array<std::size_t, 3U> peak{};
};

inline DeviceMemoryTracker& device_memory_tracker() noexcept {
    static DeviceMemoryTracker tracker;
    return tracker;
}

inline nlohmann::ordered_json device_memory_json(
    std::size_t transient_call_peak_bytes = 0U
) {
    std::size_t free_bytes = 0U;
    std::size_t total_bytes = 0U;
    check_cuda(
        cudaMemGetInfo(&free_bytes, &total_bytes),
        "performance inspect device memory"
    );
    const DeviceMemoryTracker& tracker = device_memory_tracker();
    const std::size_t persistent = tracker.peak[
        static_cast<std::size_t>(DeviceMemoryRole::persistent_input)
    ];
    const std::size_t workspace = tracker.peak[
        static_cast<std::size_t>(DeviceMemoryRole::caller_workspace)
    ];
    const std::size_t output = tracker.peak[
        static_cast<std::size_t>(DeviceMemoryRole::output)
    ];
    const std::size_t tracked_live = std::accumulate(
        tracker.live.begin(), tracker.live.end(), std::size_t{0}
    );
    const std::size_t tracked_peak =
        persistent + workspace + output + transient_call_peak_bytes;
    const std::size_t resident = total_bytes - free_bytes;
    return {
        {"persistent_input_bytes", persistent},
        {"caller_workspace_bytes", workspace},
        {"transient_call_peak_bytes", transient_call_peak_bytes},
        {"output_bytes", output},
        {"tracked_peak_bytes", tracked_peak},
        {"observed_resident_bytes", total_bytes - free_bytes},
        {"driver_context_and_pool_bytes",
            resident > tracked_live ? resident - tracked_live : 0U},
        {"free_margin_bytes", free_bytes},
        {"total_bytes", total_bytes},
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
    int repetitions = kDefaultRepetitions,
    nlohmann::ordered_json extra_timings = nlohmann::ordered_json::object(),
    std::size_t transient_call_peak_bytes = 0U
) {
    configuration["operations_per_timing_sample"] =
        measurement.operations_per_sample;
    nlohmann::ordered_json report{
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
            {"maximum_publication_noise_coefficient",
                kMaximumPublicationNoiseCoefficient},
            {"wall_semantics",
                "max(raw_host_clock, enclosing_cuda_event_interval)"},
        }},
        {"configuration", std::move(configuration)},
        {"kernel", timing_json(measurement.kernel)},
        {"public_api", timing_json(measurement.wall)},
        {"pipeline", timing_json(measurement.raw_host_clock)},
        {"device_memory", device_memory_json(transient_call_peak_bytes)},
        {"numerical_check", std::move(numerical_check)},
    };
    for (auto& [name, value] : extra_timings.items()) {
        report[name] = std::move(value);
    }
    std::cout << report.dump() << std::endl;
}

class DeviceBuffer {
public:
    DeviceBuffer(std::size_t bytes, DeviceMemoryRole role)
        : bytes_(bytes), role_(role) {
        check_cuda(cudaMalloc(&pointer_, bytes_), "performance cudaMalloc");
        DeviceMemoryTracker& tracker = device_memory_tracker();
        if (std::all_of(
                tracker.live.begin(), tracker.live.end(),
                [](std::size_t live_bytes) { return live_bytes == 0U; }
            )) {
            tracker.peak.fill(0U);
        }
        const std::size_t index = static_cast<std::size_t>(role_);
        tracker.live[index] += bytes_;
        tracker.peak[index] = std::max(tracker.peak[index], tracker.live[index]);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            static_cast<void>(cudaFree(pointer_));
            device_memory_tracker().live[static_cast<std::size_t>(role_)]
                -= bytes_;
        }
    }

    template<typename Value>
    Value* as() {
        return static_cast<Value*>(pointer_);
    }

private:
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0U;
    DeviceMemoryRole role_;
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
