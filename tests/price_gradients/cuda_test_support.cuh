// Shared public-API fixtures for selected-gradient CUDA contract tests.
#pragma once
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include <bit>
#include <cmath>
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
inline void same_or_one_ulp(float first, float second, const char* message) {
    if (first == second
        || std::nextafter(first, second) == second
        || std::nextafter(second, first) == first) {
        return;
    }
    std::cerr << message << ": " << std::hexfloat << first << " vs "
              << second << std::defaultfloat << '\n';
    throw std::runtime_error(message);
}
inline void same_within_ulps(
    float first,
    float second,
    unsigned int maximum_ulps,
    const char* message
) {
    if (!std::isfinite(first) || !std::isfinite(second)) {
        same(first, second, message);
        return;
    }
    float candidate = first;
    for (unsigned int distance = 0U;
         distance <= maximum_ulps;
         ++distance) {
        if (candidate == second) return;
        candidate = std::nextafter(candidate, second);
    }
    std::cerr << message << ": " << std::hexfloat << first << " vs "
              << second << std::defaultfloat << '\n';
    throw std::runtime_error(message);
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

template<typename Plan>
typename Plan::Preparation::Scenario central_scenario(
    const Plan& plan,
    std::size_t row
) {
    const auto indices = pg::price_row_indices(
        row, plan.construction, plan.products.size()
    );
    typename Plan::Preparation::Scenario central{};
    require(
        Plan::Preparation::make_central(
            plan.models[indices.model],
            plan.products[indices.product],
            plan.time,
            central
        ),
        "Central device-prepared test row is invalid."
    );
    return central;
}

template<pg::SensitivityOrders Orders, typename Plan>
auto sensitivity_task(
    const Plan& plan,
    std::size_t row,
    std::size_t sensitivity
) {
    using Task = pg::SensitivityTask<
        typename Plan::Preparation::Scenario,
        pg::SensitivityTraits<Orders>::node_capacity
    >;
    Task result{};
    int error = equity::price_gradients::device_preparation::valid;
    const auto central = central_scenario(plan, row);
    require(
        equity::price_gradients::device_preparation::build_sensitivity_task<
            Orders,
            typename Plan::Preparation
        >(
            central,
            plan.sensitivities[sensitivity],
            plan.time,
            result,
            error
        ),
        "Device-prepared test sensitivity is invalid."
    );
    return result;
}

template<typename Plan, typename Launcher>
Results execute(const Plan& plan, pg::LaunchConfiguration configuration, Launcher launcher, bool split = false) {
    static_assert(Plan::kDevicePreparedSensitivities);
    DeviceArray<typename Plan::Preparation::Model> models(plan.models);
    DeviceArray<typename Plan::Preparation::Product> products(plan.products);
    DeviceArray<typename Plan::SensitivitySpec> sensitivities(
        plan.sensitivities
    );
    const auto rows = plan.result_count, k = plan.sensitivity_count();
    DeviceArray<pg::SensitivityStencil<3U>> stencils(rows*k);
    DeviceArray<
        equity::price_gradients::device_preparation::Error
    > error(1U);
    DeviceArray<float> values(2U*rows + 2U*rows*k);
    const bool monte_carlo =
        configuration.method == pg::PricingMethod::monte_carlo;
    const typename Plan::DeviceInputs inputs{
        models.data, models.count,
        products.data, products.count,
        sensitivities.data, sensitivities.count,
    };
    const typename Plan::StencilOutputs stencil_outputs{
        stencils.data, stencils.count, error.data
    };
    pg::Outputs outputs{
        values.data, monte_carlo ? values.data+rows : nullptr,
        values.data+2U*rows,
        monte_carlo ? values.data+2U*rows+rows*k : nullptr,
        rows, rows*k
    };
    configuration.result_count = rows;
    bool rejected = false;
    if (monte_carlo) {
        configuration.sensitivity_batch_size = 1U;
        configuration.block_count = rows*(k == 0U ? 1U : k);
    }
    const auto invoke = [&](const auto& launch_configuration) {
        launcher(
            plan, inputs, stencil_outputs, launch_configuration, outputs
        );
    };
    if (split && rows > 1U) {
        configuration.result_count = rows-1U;
        if (monte_carlo) {
            configuration.block_count =
                configuration.result_count*(k == 0U ? 1U : k);
        }
        invoke(configuration);
        configuration.result_offset = rows-1U;
        configuration.result_count = 1U;
        if (monte_carlo) {
            configuration.block_count = k == 0U ? 1U : k;
        }
        invoke(configuration);
    } else {
        invoke(configuration);
    }
    if (monte_carlo) {
        for (unsigned int width : {0U,3U,5U}) {
            rejected = false;
            auto bad = configuration;
            bad.sensitivity_batch_size = width;
            try { invoke(bad); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected,"Invalid sensitivity batch size accepted.");
        }
        rejected = false;
        auto bad_configuration = configuration;
        bad_configuration.block_count =
            bad_configuration.result_count*(k == 0U ? 1U : k)+1U;
        try { invoke(bad_configuration); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected,"Excess sensitivity blocks accepted.");
    }
    const auto host = values.read();
    const auto preparation_error = error.read();
    require(
        preparation_error[0].code == 0,
        "Device sensitivity preparation failed."
    );
    Results result{
        {host.begin(), host.begin()+rows},
        {},
        {host.begin()+2U*rows, host.begin()+2U*rows+rows*k},
        {}
    };
    if (monte_carlo) {
        result.price_error.assign(host.begin()+rows, host.begin()+2U*rows);
        result.gradient_error.assign(
            host.begin()+2U*rows+rows*k, host.end()
        );
    }
    for (float value : result.price) {
        require(std::isfinite(value), "Non-finite price.");
    }
    for (float value : result.gradient) {
        require(std::isfinite(value), "Non-finite gradient.");
    }
    for (float value : result.gradient_error) {
        require(
            std::isfinite(value) && value >= 0,
            "Invalid gradient error."
        );
    }
    rejected = false;
    try {
        auto bad = outputs;
        bad.gradients = bad.prices;
        launcher(plan, inputs, stencil_outputs, configuration, bad);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (k) require(rejected, "Aliased gradient outputs were accepted.");
    rejected = false;
    try {
        auto bad = inputs;
        bad.model_capacity = 0U;
        launcher(plan, bad, stencil_outputs, configuration, outputs);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Undersized model buffer was accepted.");
    return result;
}

// Compare every selected prefix with the full-selection reference, including
// price-only delegation, grid-stride reuse and global row offsets.
template<typename Prepare, typename Launcher>
void selected_prefixes(
    const pg::PriceGradientConfiguration& full,
    const Results& reference,
    pg::LaunchConfiguration launch,
    Prepare prepare,
    Launcher launcher
) {
    launch.sensitivity_batch_size = 1U;
    for (std::size_t k = 0; k <= full.sensitivities.size(); ++k) {
        auto selection = full;
        selection.sensitivities.resize(k);
        const auto plan = prepare(selection);
        launch.block_count = plan.result_count * (k == 0U ? 1U : k);
        const auto candidate = execute(plan,launch,launcher);
        for (std::size_t row = 0; row < plan.result_count; ++row) {
            same(candidate.price[row],reference.price[row],
                 "Selected prefix changed price");
            same(candidate.price_error[row],reference.price_error[row],
                 "Selected prefix changed price error");
            for (std::size_t i = 0; i < k; ++i) {
                same(candidate.gradient[row*k+i],
                     reference.gradient[row*full.sensitivities.size()+i],
                     "Selected prefix changed gradient");
                same(candidate.gradient_error[row*k+i],
                     reference.gradient_error[
                         row*full.sensitivities.size()+i
                     ],
                     "Selected prefix changed gradient error");
            }
        }
        // One x-block per sensitivity and split launches must write exactly
        // the same rows and columns as the fully populated grid.
        launch.block_count = k == 0U ? 1U : k;
        const auto strided = execute(plan,launch,launcher,true);
        require(
            strided.price == candidate.price
                && strided.price_error == candidate.price_error
                && strided.gradient == candidate.gradient
                && strided.gradient_error == candidate.gradient_error,
            "Grid-stride or split launch changed outputs."
        );
    }
}

}  // namespace price_gradient_test
