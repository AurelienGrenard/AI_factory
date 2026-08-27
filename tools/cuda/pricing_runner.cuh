// Shared RAII execution pipeline for offline CUDA price generators.
#pragma once

#include "common/check_cuda.cuh"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace ai_factory::workbench::offline::cuda {

// Own one contiguous device allocation. Construction, copies and destruction
// are deliberately kept out of catalog recipes.
template<class Value>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ != 0U) {
            check_cuda(
                cudaMalloc(&data_, count_ * sizeof(Value)),
                "offline CUDA buffer allocation"
            );
        }
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0U)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        if (data_ != nullptr) cudaFree(data_);
        data_ = std::exchange(other.data_, nullptr);
        count_ = std::exchange(other.count_, 0U);
        return *this;
    }

    Value* data() noexcept { return data_; }
    const Value* data() const noexcept { return data_; }

    void copy_from(const Value* source) {
        if (count_ == 0U) return;
        check_cuda(
            cudaMemcpy(
                data_, source, count_ * sizeof(Value), cudaMemcpyHostToDevice
            ),
            "offline CUDA input copy"
        );
    }

    void copy_to(Value* destination) const {
        if (count_ == 0U) return;
        check_cuda(
            cudaMemcpy(
                destination, data_, count_ * sizeof(Value),
                cudaMemcpyDeviceToHost
            ),
            "offline CUDA output copy"
        );
    }

private:
    Value* data_ = nullptr;
    std::size_t count_ = 0U;
};

class Event {
public:
    Event() {
        check_cuda(cudaEventCreate(&event_), "offline CUDA event creation");
    }

    ~Event() {
        if (event_ != nullptr) cudaEventDestroy(event_);
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

template<class... Values>
struct HostInputs {
    std::tuple<const std::vector<Values>*...> values;
};

template<class... Containers>
auto inputs(const Containers&... containers) {
    return HostInputs<typename Containers::value_type...>{
        std::tuple{&containers...}
    };
}

template<bool WithStandardErrors, class... Values>
class Execution {
public:
    Execution(const HostInputs<Values...>& host, std::size_t result_count)
        : inputs_(make_buffers(host, std::index_sequence_for<Values...>{})),
          prices_(result_count),
          standard_errors_(WithStandardErrors ? result_count : 0U) {
        copy_inputs(host, std::index_sequence_for<Values...>{});
    }

    template<std::size_t Index>
    const auto* input() const noexcept {
        return std::get<Index>(inputs_).data();
    }

    float* prices() noexcept { return prices_.data(); }

    float* standard_errors() noexcept {
        static_assert(
            WithStandardErrors,
            "Analytical executions do not allocate standard errors."
        );
        return standard_errors_.data();
    }

    void copy_prices_to(std::vector<float>& destination) const {
        prices_.copy_to(destination.data());
    }

    void copy_standard_errors_to(std::vector<float>& destination) const {
        static_assert(
            WithStandardErrors,
            "Analytical executions do not allocate standard errors."
        );
        standard_errors_.copy_to(destination.data());
    }

private:
    template<std::size_t... Indices>
    static auto make_buffers(
        const HostInputs<Values...>& host,
        std::index_sequence<Indices...>
    ) {
        return std::tuple<DeviceBuffer<Values>...>{
            DeviceBuffer<Values>(std::get<Indices>(host.values)->size())...
        };
    }

    template<std::size_t... Indices>
    void copy_inputs(
        const HostInputs<Values...>& host,
        std::index_sequence<Indices...>
    ) {
        (
            std::get<Indices>(inputs_).copy_from(
                std::get<Indices>(host.values)->data()
            ),
            ...
        );
    }

    std::tuple<DeviceBuffer<Values>...> inputs_;
    DeviceBuffer<float> prices_;
    DeviceBuffer<float> standard_errors_;
};

struct AnalyticalRun {
    std::vector<float> prices;
    double wall_seconds = 0.0;
    double kernel_seconds = 0.0;
};

struct MonteCarloRun : AnalyticalRun {
    std::vector<float> standard_errors;
};

template<bool WithStandardErrors, class... Values, class Warmup, class Launch>
auto run(
    const HostInputs<Values...>& host_inputs,
    std::size_t result_count,
    Warmup&& warmup,
    Launch&& launch
) {
    using Run = std::conditional_t<
        WithStandardErrors, MonteCarloRun, AnalyticalRun
    >;
    Run result;
    result.prices.resize(result_count);
    if constexpr (WithStandardErrors) {
        result.standard_errors.resize(result_count);
    }

    const auto wall_start = std::chrono::steady_clock::now();
    Execution<WithStandardErrors, Values...> execution(
        host_inputs, result_count
    );
    std::invoke(std::forward<Warmup>(warmup), execution);
    check_cuda(cudaDeviceSynchronize(), "offline CUDA warmup");

    Event start;
    Event stop;
    check_cuda(cudaEventRecord(start.get()), "offline CUDA timer start");
    std::invoke(std::forward<Launch>(launch), execution);
    check_cuda(cudaEventRecord(stop.get()), "offline CUDA timer stop");
    check_cuda(cudaEventSynchronize(stop.get()), "offline CUDA timer wait");
    float kernel_milliseconds = 0.0f;
    check_cuda(
        cudaEventElapsedTime(
            &kernel_milliseconds, start.get(), stop.get()
        ),
        "offline CUDA elapsed time"
    );
    result.kernel_seconds =
        static_cast<double>(kernel_milliseconds) * 1.0e-3;

    execution.copy_prices_to(result.prices);
    if constexpr (WithStandardErrors) {
        execution.copy_standard_errors_to(result.standard_errors);
    }
    result.wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start
    ).count();
    return result;
}

template<class... Values, class Warmup, class Launch>
MonteCarloRun run_monte_carlo(
    const HostInputs<Values...>& host_inputs,
    std::size_t result_count,
    Warmup&& warmup,
    Launch&& launch
) {
    return run<true>(
        host_inputs,
        result_count,
        std::forward<Warmup>(warmup),
        std::forward<Launch>(launch)
    );
}

template<class... Values, class Warmup, class Launch>
AnalyticalRun run_analytical(
    const HostInputs<Values...>& host_inputs,
    std::size_t result_count,
    Warmup&& warmup,
    Launch&& launch
) {
    return run<false>(
        host_inputs,
        result_count,
        std::forward<Warmup>(warmup),
        std::forward<Launch>(launch)
    );
}

}  // namespace ai_factory::workbench::offline::cuda
