// Shared fixed-workload public MC gradient benchmark and output capture.
#pragma once
#include "common/price_gradients/launch.cuh"
#include "tests/performance/benchmark_support.cuh"
#include <bit>
namespace gradient_benchmark {
using namespace ai_factory::workbench;
namespace pg=price_gradients;
template<typename T> struct DeviceBuffer {
    T* data = nullptr;
    std::size_t count;
    explicit DeviceBuffer(std::size_t size) : count(size) {
        if (size) check_cuda(cudaMalloc(&data, size*sizeof(T)), "benchmark allocate");
    }
    explicit DeviceBuffer(const std::vector<T>& source) : DeviceBuffer(source.size()) {
        if (count) check_cuda(cudaMemcpy(data, source.data(), count*sizeof(T), cudaMemcpyHostToDevice), "benchmark upload");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template<typename Plan, typename Launcher>
void measure(const char* model, const Plan& plan, Launcher launcher, unsigned threads, unsigned width) {
    const auto rows = plan.result_count, k = plan.sensitivity_count();
    DeviceBuffer<typename Plan::ScenarioType> scenarios(plan.scenarios);
    DeviceBuffer<pg::Stencil> stencils(plan.stencils);
    DeviceBuffer<float> values(2U*rows*(1U+k));
    const pg::DeviceInputs<typename Plan::ScenarioType> inputs{scenarios.data,scenarios.count,stencils.data,stencils.count};
    const pg::Outputs outputs{values.data,values.data+rows,values.data+2U*rows,values.data+2U*rows+rows*k,rows,rows*k};
    pg::LaunchConfiguration config{pg::PricingMethod::monte_carlo,0U,rows,8192U,threads,rows,719U};
    config.sensitivity_batch_size = width;
    config.block_count = rows*pg::sensitivity_batch_count(k,width);
    // BS kernels are too short for stable ungrouped event windows. Keep the
    // same declared grouping for reference and candidate; report per operation.
    const std::size_t operations = std::string_view(model)=="black_scholes" ? 64U : 1U;
    const auto timing = performance::measure_cuda([&] { launcher(plan,inputs,config,outputs); },
        performance::kDefaultWarmups,performance::kDefaultRepetitions,operations);
    std::vector<float> host(values.count);
    check_cuda(cudaMemcpy(host.data(),values.data,host.size()*sizeof(float),cudaMemcpyDeviceToHost),"benchmark download");
    std::vector<std::uint32_t> bits;
    for (float value : host) {
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite benchmark output.");
        bits.push_back(std::bit_cast<std::uint32_t>(value));
    }
    nlohmann::ordered_json report{
        {"model",model},{"rows",rows},{"k",k},{"batch_size",width},{"threads",threads},
        {"paths",config.paths_per_price},{"seed",config.base_seed},{"operations_per_sample",operations},
        {"warmups",performance::kDefaultWarmups},{"repetitions",performance::kDefaultRepetitions},
        {"kernel",performance::timing_json(timing.kernel)},
        {"public_api",performance::timing_json(timing.wall)},
        {"raw_host_clock",performance::timing_json(timing.raw_host_clock)},
        {"environment",performance::environment_json()},{"output_bits",bits}};
    std::cout << report.dump() << std::endl;
}

}  // namespace gradient_benchmark
