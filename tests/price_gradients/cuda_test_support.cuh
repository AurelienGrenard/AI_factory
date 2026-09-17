// Shared public-API fixtures for selected-gradient CUDA contract tests.
#pragma once
#include "common/price_gradients/launch.cuh"
#include <bit>
#include <iostream>
#include <vector>

namespace price_gradient_test {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
inline void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
inline void same(float a, float b, const char* message) {
    if (std::bit_cast<std::uint32_t>(a) != std::bit_cast<std::uint32_t>(b)) {
        std::cerr << message << ": " << std::hexfloat << a << " vs " << b << std::defaultfloat << '\n';
        throw std::runtime_error(message);
    }
}
template<typename T> struct DeviceArray {
    T* data = nullptr;
    std::size_t count;
    explicit DeviceArray(std::size_t size) : count(size) {
        if (size) {
            check_cuda(cudaMalloc(&data, size*sizeof(T)), "test allocation");
            check_cuda(cudaMemset(data, 0, size*sizeof(T)), "test initialization");
        }
    }
    explicit DeviceArray(const std::vector<T>& values) : DeviceArray(values.size()) {
        if (count) check_cuda(cudaMemcpy(data, values.data(), count*sizeof(T), cudaMemcpyHostToDevice), "test upload");
    }
    ~DeviceArray() { cudaFree(data); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    std::vector<T> read() const {
        std::vector<T> result(count);
        if (count) check_cuda(cudaMemcpy(result.data(), data, count*sizeof(T), cudaMemcpyDeviceToHost), "test download");
        return result;
    }
};
struct Results { std::vector<float> price, price_error, gradient, gradient_error; };
template<typename Plan, typename Launcher>
Results execute(const Plan& plan, pg::LaunchConfiguration configuration, Launcher launcher, bool split = false) {
    DeviceArray<typename Plan::ScenarioType> scenarios(plan.scenarios);
    DeviceArray<pg::Stencil> stencils(plan.stencils);
    const auto rows = plan.result_count, k = plan.sensitivity_count();
    DeviceArray<float> values(2U*rows + 2U*rows*k);
    pg::DeviceInputs<typename Plan::ScenarioType> inputs{scenarios.data, scenarios.count, stencils.data, stencils.count};
    pg::Outputs outputs{values.data, values.data+rows, values.data+2U*rows, values.data+2U*rows+rows*k, rows, rows*k};
    configuration.result_count = rows;
    if (split && rows > 1U) {
        configuration.result_count = rows-1U;
        configuration.block_count = 1U;
        launcher(plan, inputs, configuration, outputs);
        configuration.result_offset = rows-1U;
        configuration.result_count = 1U;
        launcher(plan, inputs, configuration, outputs);
    } else launcher(plan, inputs, configuration, outputs);
    if (configuration.method == pg::PricingMethod::monte_carlo) {
        for (unsigned int width : {0U,3U,5U}) {
            bool rejected = false;
            auto bad = configuration; bad.sensitivity_batch_size = width;
            try { launcher(plan,inputs,bad,outputs); } catch (const std::invalid_argument&) { rejected = true; }
            require(rejected,"Invalid sensitivity batch size accepted.");
        }
        bool rejected = false;
        auto bad = configuration;
        bad.block_count = pg::gradient_task_count(bad.result_count,
            pg::sensitivity_batch_count(k,bad.sensitivity_batch_size))+1U;
        try { launcher(plan,inputs,bad,outputs); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected,"Excess row/batch blocks accepted.");
    }
    const auto host = values.read();
    const bool mc = configuration.method == pg::PricingMethod::monte_carlo;
    Results result{{host.begin(), host.begin()+rows}, {}, {host.begin()+2U*rows, host.begin()+2U*rows+rows*k}, {}};
    if (mc) {
        result.price_error.assign(host.begin()+rows, host.begin()+2U*rows);
        result.gradient_error.assign(host.begin()+2U*rows+rows*k, host.end());
    }
    for (float value : result.price) require(std::isfinite(value), "Non-finite price.");
    for (float value : result.gradient) require(std::isfinite(value), "Non-finite gradient.");
    for (float value : result.gradient_error) require(std::isfinite(value) && value >= 0, "Invalid gradient error.");
    bool rejected = false;
    try { auto bad = outputs; bad.gradients = bad.prices; launcher(plan, inputs, configuration, bad); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (k) require(rejected, "Aliased gradient outputs were accepted.");
    rejected = false;
    try { auto bad = inputs; bad.scenario_capacity = 0; launcher(plan, bad, configuration, outputs); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Undersized scenario buffer was accepted.");
    return result;
}

// Compare every prefix and tail width with the full-selection reference, including
// price-only, multiple blocks per row, grid-stride reuse and global row offsets.
template<typename Prepare, typename Launcher>
void batching(const pg::PriceGradientConfiguration& full, const Results& reference,
              pg::LaunchConfiguration launch, Prepare prepare, Launcher launcher) {
    for (unsigned int width : {1U,2U,4U}) {
        launch.sensitivity_batch_size = width;
        for (std::size_t k = 0; k <= full.sensitivities.size(); ++k) {
            auto selection = full; selection.sensitivities.resize(k);
            const auto plan = prepare(selection);
            launch.block_count = plan.result_count*pg::sensitivity_batch_count(k,width);
            const auto candidate = execute(plan,launch,launcher);
            for (std::size_t row = 0; row < plan.result_count; ++row) {
                same(candidate.price[row],reference.price[row],"Batch width changed price");
                same(candidate.price_error[row],reference.price_error[row],"Batch width changed price error");
                for (std::size_t i = 0; i < k; ++i) {
                    same(candidate.gradient[row*k+i],reference.gradient[row*full.sensitivities.size()+i],"Batch width/tail changed gradient");
                    same(candidate.gradient_error[row*k+i],reference.gradient_error[row*full.sensitivities.size()+i],"Batch width/tail changed gradient error");
                }
            }
            // A single grid-striding block and split launches must write exactly
            // the same rows/columns as the fully populated row/batch grid.
            launch.block_count = 1U;
            const auto strided = execute(plan,launch,launcher,true);
            require(strided.price == candidate.price && strided.price_error == candidate.price_error
                && strided.gradient == candidate.gradient && strided.gradient_error == candidate.gradient_error,
                "Grid-stride/split batching changed outputs.");
        }
    }
}

}  // namespace price_gradient_test
