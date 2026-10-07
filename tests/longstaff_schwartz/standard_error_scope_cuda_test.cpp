// Show that the in-sample LSM cashflow SE can be zero while the fit changes by seed.
#include "common/check_cuda.cuh"
#include "common/longstaff_schwartz/host_progress.cuh"
#include "common/longstaff_schwartz/price_gradients/execution_plan.cuh"
#include "model/equity/markovian/black_scholes/product/american_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    namespace bs = model::equity::black_scholes;
    int count = 0;
    const auto available = cudaGetDeviceCount(&count);
    if (available == cudaErrorNoDevice
        || available == cudaErrorInsufficientDriver || count == 0) return 77;
    check_cuda(available, "LSM scope cudaGetDeviceCount");

    namespace lspg = longstaff_schwartz::price_gradients;
    const std::size_t grid_y_limit = 65535U;
    const std::size_t last_valid_batch = lspg::maximum_batch_size(4U, grid_y_limit);
    if (last_valid_batch != 16383U)
        throw std::runtime_error("Unexpected LSM gradient batch limit");
    lspg::validate_task_grid(last_valid_batch, 4U, grid_y_limit);
    bool rejected = false;
    try {
        lspg::validate_task_grid(last_valid_batch + 1U, 4U, grid_y_limit);
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("LSM gradient gridDim.y overflow accepted");

    const bs::ModelParameters model{1.0f, 0.08f, 0.0f, 0.15f};
    product::AmericanOptionParameters option{1.09f, 252U, 21U};
    bs::ModelParameters* device_model = nullptr;
    product::AmericanOptionParameters* device_option = nullptr;
    float* device_price = nullptr;
    float* device_error = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "LSM scope model malloc");
    check_cuda(cudaMalloc(&device_option, sizeof(option)), "LSM scope option malloc");
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "LSM scope price malloc");
    check_cuda(cudaMalloc(&device_error, sizeof(float)), "LSM scope error malloc");
    try {
        std::size_t zero_error_count = 0U;
        std::size_t positive_error_count = 0U;
        float minimum_price = INFINITY;
        float maximum_price = -INFINITY;
        check_cuda(cudaMemcpy(device_model, &model, sizeof(model),
                              cudaMemcpyHostToDevice), "LSM scope model copy");
        check_cuda(cudaMemcpy(device_option, &option, sizeof(option),
                              cudaMemcpyHostToDevice), "LSM scope option copy");
        for (std::uint64_t seed = 1U; seed <= 16U; ++seed) {
            struct Progress { std::size_t calls = 0U; std::size_t completed = 0U; } progress;
            auto callback = [](std::size_t completed, void* context) noexcept {
                auto& value = *static_cast<Progress*>(context);
                ++value.calls;
                value.completed = completed;
            };
            longstaff_schwartz::ScopedHostProgress active(callback, &progress);
            auto execution = bs::launch_black_scholes_american_option_cuda<
                OptionSide::put>(
                    device_model, 1U, &option, device_option, 1U,
                    PriceConstruction::Aligned, 1U, 1024U, 1.0f / 252.0f,
                    128U, 8U, seed, device_price, device_error
                );
            longstaff_schwartz::validate_regression_diagnostics(
                execution, "LSM standard-error scope"
            );
            float price = 0.0f;
            float error = 0.0f;
            check_cuda(cudaMemcpy(&price, device_price, sizeof(float),
                                  cudaMemcpyDeviceToHost), "LSM scope price download");
            check_cuda(cudaMemcpy(&error, device_error, sizeof(float),
                                  cudaMemcpyDeviceToHost), "LSM scope error download");
            if (!std::isfinite(price) || !std::isfinite(error) || error < 0.0f)
                throw std::runtime_error("Invalid LSM scope output");
            if (progress.calls != 1U || progress.completed != 1U)
                throw std::runtime_error("LSM host progress callback missed completed batch");
            minimum_price = std::min(minimum_price, price);
            maximum_price = std::max(maximum_price, price);
            if (error == 0.0f) {
                if (std::abs(price - (option.strike - model.spot)) > 1.0e-6f)
                    throw std::runtime_error("Zero SE must be t0 immediate exercise");
                ++zero_error_count;
            } else {
                ++positive_error_count;
            }
            std::cout << seed << ' ' << price << ' ' << error << '\n';
        }
        if (zero_error_count == 0U || positive_error_count == 0U
            || maximum_price - minimum_price <= 1.0e-4f)
            throw std::runtime_error("Multi-seed fit uncertainty was not exposed");
        std::cout << "zero_error=" << zero_error_count
                  << " positive_error=" << positive_error_count
                  << " price_range=" << maximum_price - minimum_price << '\n';
    } catch (...) {
        cudaFree(device_error);
        cudaFree(device_price);
        cudaFree(device_option);
        cudaFree(device_model);
        throw;
    }
    cudaFree(device_error);
    cudaFree(device_price);
    cudaFree(device_option);
    cudaFree(device_model);
}
