// Shared fixed-workload public MC gradient benchmark and output capture.
#pragma once
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
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
void measure(
    const char* model,
    const Plan& plan,
    Launcher launcher,
    unsigned int threads
) {
    static_assert(Plan::kDevicePreparedSensitivities);
    const auto rows = plan.result_count, k = plan.sensitivity_count();
    DeviceBuffer<typename Plan::Preparation::Model> models(plan.models);
    DeviceBuffer<typename Plan::Preparation::Product> products(plan.products);
    DeviceBuffer<typename Plan::SensitivitySpec> sensitivities(
        plan.sensitivities
    );
    DeviceBuffer<pg::SensitivityStencil<3U>> stencils(rows*k);
    const std::vector<
        equity::price_gradients::device_preparation::Error
    > initial_error(1U);
    DeviceBuffer<
        equity::price_gradients::device_preparation::Error
    > error(initial_error);
    DeviceBuffer<float> values(2U*rows*(1U+k));
    const typename Plan::DeviceInputs inputs{
        models.data, models.count,
        products.data, products.count,
        sensitivities.data, sensitivities.count,
    };
    const typename Plan::StencilOutputs stencil_outputs{
        stencils.data, stencils.count, error.data
    };
    const pg::Outputs outputs{
        values.data,values.data+rows,
        values.data+2U*rows,
        values.data+2U*rows+rows*k,
        rows,rows*k
    };
    pg::LaunchConfiguration config{
        pg::PricingMethod::monte_carlo,0U,rows,8192U,threads,
        rows*(k == 0U ? 1U : k),719U,1U
    };
    const auto timing = performance::measure_cuda(
        [&] {
            launcher(
                plan,inputs,stencil_outputs,config,outputs
            );
        },
        performance::kDefaultWarmups,
        performance::kDefaultRepetitions,
        1U
    );
    std::vector<float> host(values.count);
    check_cuda(
        cudaMemcpy(
            host.data(),values.data,host.size()*sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "benchmark download"
    );
    std::vector<std::uint32_t> bits;
    for (float value : host) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite benchmark output.");
        }
        bits.push_back(std::bit_cast<std::uint32_t>(value));
    }
    nlohmann::ordered_json report{
        {"model",model},{"rows",rows},{"k",k},{"batch_size",1U},
        {"threads",threads},{"paths",config.paths_per_price},
        {"seed",config.base_seed},{"operations_per_sample",1U},
        {"warmups",performance::kDefaultWarmups},
        {"repetitions",performance::kDefaultRepetitions},
        {"kernel",performance::timing_json(timing.kernel)},
        {"public_api",performance::timing_json(timing.wall)},
        {"raw_host_clock",performance::timing_json(timing.raw_host_clock)},
        {"environment",performance::environment_json()},
        {"output_bits",bits}
    };
    std::cout << report.dump() << std::endl;
}

template<typename Plan, typename Launcher>
void measure_diagonal(
    const char* model,
    const Plan& plan,
    Launcher launcher,
    unsigned int threads
) {
    static_assert(Plan::kDevicePreparedSensitivities);
    const auto rows = plan.result_count;
    const auto k = plan.sensitivity_count();
    DeviceBuffer<typename Plan::Preparation::Model> models(plan.models);
    DeviceBuffer<typename Plan::Preparation::Product> products(plan.products);
    DeviceBuffer<typename Plan::SensitivitySpec> sensitivities(
        plan.sensitivities
    );
    DeviceBuffer<pg::SensitivityStencil<4U>> stencils(rows * k);
    const std::vector<
        equity::price_gradients::device_preparation::Error
    > initial_error(1U);
    DeviceBuffer<
        equity::price_gradients::device_preparation::Error
    > error(initial_error);
    DeviceBuffer<float> values(2U * rows + 4U * rows * k);
    const typename Plan::DeviceInputs inputs{
        models.data, models.count,
        products.data, products.count,
        sensitivities.data, sensitivities.count,
    };
    const typename Plan::DiagonalStencilOutputs stencil_outputs{
        stencils.data, stencils.count, error.data
    };
    const pg::SensitivityOutputs outputs{
        values.data,
        values.data + rows,
        values.data + 2U * rows,
        values.data + 2U * rows + rows * k,
        values.data + 2U * rows + 2U * rows * k,
        values.data + 2U * rows + 3U * rows * k,
        rows,
        rows * k,
    };
    const pg::LaunchConfiguration configuration{
        pg::PricingMethod::monte_carlo,
        0U,
        rows,
        8192U,
        threads,
        rows * k,
        719U,
        1U,
    };
    const auto timing = performance::measure_cuda(
        [&] {
            launcher(
                plan,
                inputs,
                stencil_outputs,
                configuration,
                outputs
            );
        },
        performance::kDefaultWarmups,
        performance::kDefaultRepetitions,
        1U
    );
    std::vector<float> host(values.count);
    check_cuda(
        cudaMemcpy(
            host.data(),
            values.data,
            host.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "diagonal sensitivity benchmark download"
    );
    std::vector<std::uint32_t> bits;
    bits.reserve(host.size());
    for (const float value : host) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite diagonal benchmark output.");
        }
        bits.push_back(std::bit_cast<std::uint32_t>(value));
    }
    const nlohmann::ordered_json report{
        {"model", model},
        {"orders", "first_and_diagonal_second"},
        {"rows", rows},
        {"k", k},
        {"batch_size", 1U},
        {"threads", threads},
        {"paths", configuration.paths_per_price},
        {"seed", configuration.base_seed},
        {"operations_per_sample", 1U},
        {"warmups", performance::kDefaultWarmups},
        {"repetitions", performance::kDefaultRepetitions},
        {"kernel", performance::timing_json(timing.kernel)},
        {"public_api", performance::timing_json(timing.wall)},
        {"raw_host_clock", performance::timing_json(timing.raw_host_clock)},
        {"environment", performance::environment_json()},
        {"output_bits", bits},
    };
    std::cout << report.dump() << std::endl;
}

}  // namespace gradient_benchmark
