// Gaussian rough FFT graph: shared spectra, CRN and central pricing parity.
#include "common/check_cuda.cuh"
#include "common/volterra/hybrid_fft_workspace.cuh"
#ifndef ROUGH_GAUSSIAN_TEST_MODEL
#define ROUGH_GAUSSIAN_TEST_MODEL 0
#endif
#if ROUGH_GAUSSIAN_TEST_MODEL == 0
#include "model/equity/rough/rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/rough_bergomi/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_bergomi/product/european_option_price_gradients.cuh"
#elif ROUGH_GAUSSIAN_TEST_MODEL == 1
#include "model/equity/rough/rough_sabr/product/european_option.cuh"
#include "model/equity/rough/rough_sabr/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_sabr/product/european_option_price_gradients.cuh"
#elif ROUGH_GAUSSIAN_TEST_MODEL == 2
#include "model/equity/rough/rough_stein_stein/product/european_option.cuh"
#include "model/equity/rough/rough_stein_stein/product/european_option_price_delta.cuh"
#include "model/equity/rough/rough_stein_stein/product/european_option_price_gradients.cuh"
#elif ROUGH_GAUSSIAN_TEST_MODEL == 3
#include "model/equity/rough/log_modulated_rough_bergomi/product/european_option.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/european_option_price_delta.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/product/european_option_price_gradients.cuh"
#else
#error "Unknown Gaussian rough test model"
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace mcpg = monte_carlo::price_gradients;
#if ROUGH_GAUSSIAN_TEST_MODEL == 0
namespace rb = model::equity::rough_bergomi;
#elif ROUGH_GAUSSIAN_TEST_MODEL == 1
namespace rb = model::equity::rough_sabr;
#elif ROUGH_GAUSSIAN_TEST_MODEL == 2
namespace rb = model::equity::rough_stein_stein;
#else
namespace rb = model::equity::log_modulated_rough_bergomi;
#endif

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<typename T>
struct DeviceBuffer {
    T* value = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(&value, count * sizeof(T)), "Gaussian graph allocate");
    }
    ~DeviceBuffer() { if (value) cudaFree(value); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* source, std::size_t count) {
        check_cuda(cudaMemcpy(value, source, count * sizeof(T),
                              cudaMemcpyHostToDevice), "Gaussian graph upload");
    }
    void download(T* target, std::size_t count) const {
        check_cuda(cudaMemcpy(target, value, count * sizeof(T),
                              cudaMemcpyDeviceToHost), "Gaussian graph download");
    }
};
}

int main() {
    int count = 0;
    const auto availability = cudaGetDeviceCount(&count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver || count == 0)
        return 77;
    check_cuda(availability, "Gaussian graph cudaGetDeviceCount");
    constexpr std::size_t paths = 513U;
    constexpr std::uint64_t seed = 932000001U;
#if ROUGH_GAUSSIAN_TEST_MODEL == 0
    const std::vector<rb::ModelParameters> models{
        {1.0f, .02f, .01f, .04f, .8f, .10f, -.7f}
    };
#elif ROUGH_GAUSSIAN_TEST_MODEL == 1
    const std::vector<rb::ModelParameters> models{
        {1.0f, .02f, .01f, .04f, .8f, .10f, -.7f, .85f}
    };
#elif ROUGH_GAUSSIAN_TEST_MODEL == 2
    const std::vector<rb::ModelParameters> models{
        {1.0f, .02f, .01f, .20f, .30f, .40f, .10f, -.7f}
    };
#else
    const std::vector<rb::ModelParameters> models{
        {1.0f, .02f, .01f, .04f, .8f, .10f, -.7f, 1.0f, 2.0f}
    };
#endif
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 252U}, {1.1f, 252U}
    };
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.01f, pg::BumpScale::relative}},
        {"model.hurst_exponent", {.001f, pg::BumpScale::absolute}},
        {"product.strike", {.01f, pg::BumpScale::relative}},
        {"product.maturity_years", {1.0f / 504.0f, pg::BumpScale::absolute}}
    }};
    const auto prepare_host = [&] {
#if ROUGH_GAUSSIAN_TEST_MODEL == 0
        return rb::prepare_rough_bergomi_european_option_sensitivities(
#elif ROUGH_GAUSSIAN_TEST_MODEL == 1
        return rb::prepare_rough_sabr_european_option_sensitivities(
#elif ROUGH_GAUSSIAN_TEST_MODEL == 2
        return rb::prepare_rough_stein_stein_european_option_sensitivities(
#else
        return rb::prepare_log_modulated_rough_bergomi_european_option_sensitivities(
#endif
            models, products, PriceConstruction::CartesianProduct,
            {1.0f / 504.0f, 2U}, selection,
            {pg::SensitivityOrders::first_and_second}
        );
    };
    const auto host = prepare_host();
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, 2U, paths, 256U, 2U, seed
    };
    using Graph = rb::EuropeanOptionNodeGraph<
        OptionSide::call, pg::SensitivityOrders::first_and_second, 4U
    >;
    Graph::Cache cache;
    const auto execution = Graph::plan(host, cache, launch);
    require(cache.entries().size() == 3U,
            "Spot and strike bumps duplicated a Gaussian kernel spectrum.");
    const auto row = volterra::price_gradients::
        prepare_gaussian_fft_graph_row<
            pg::SensitivityOrders::first_and_second, 4U
        >(host, 1U, cache);
    require(row.kernel_indices[0U] == row.kernel_indices[1U]
            && row.kernel_indices[0U] == row.kernel_indices[5U],
            "Spot/strike nodes must share the central FFT.");

    DeviceBuffer<unsigned char> workspace(execution.bytes);
    DeviceBuffer<float> prices(2U), errors(2U), gradients(8U),
        gradient_errors(8U), hessians(8U), hessian_errors(8U);
    DeviceBuffer<pg::SensitivityStencil<4U>> stencils(8U);
    pg::SensitivityOutputs outputs{
        prices.value, errors.value, gradients.value,
        gradient_errors.value, hessians.value, hessian_errors.value,
        2U, 8U
    };
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs{
        stencils.value, 8U, nullptr
    };
    Graph::launch(host, cache, launch, outputs, stencil_outputs,
                  execution, workspace.value, execution.bytes);
    check_cuda(cudaDeviceSynchronize(), "Gaussian graph sync");
    float graph_prices[2]{}, graph_errors[2]{}, graph_gradients[8]{};
    prices.download(graph_prices, 2U);
    errors.download(graph_errors, 2U);
    gradients.download(graph_gradients, 8U);
    for (std::size_t i = 0U; i < 2U; ++i)
        require(std::isfinite(graph_prices[i])
                && std::isfinite(graph_errors[i])
                && graph_errors[i] > 0.0f,
                "Gaussian graph price moments are invalid.");
    for (float value : graph_gradients)
        require(std::isfinite(value), "Gaussian graph gradient is invalid.");

    DeviceBuffer<rb::ModelParameters> device_models(1U);
    DeviceBuffer<product::EuropeanOptionParameters> device_products(2U);
    DeviceBuffer<float> reference_prices(2U), reference_errors(2U);
    device_models.upload(models.data(), 1U);
    device_products.upload(products.data(), 2U);
    constexpr std::size_t chunk = 256U;
    const auto reference_bytes =
        volterra::required_hybrid_fft_workspace_bytes(504U, paths, chunk);
    DeviceBuffer<unsigned char> reference_workspace(reference_bytes);
    for (std::size_t row_index = 0U; row_index < 2U; ++row_index) {
#if ROUGH_GAUSSIAN_TEST_MODEL == 0
        rb::launch_rough_bergomi_european_option_cuda<OptionSide::call>(
#elif ROUGH_GAUSSIAN_TEST_MODEL == 1
        rb::launch_rough_sabr_european_option_cuda<OptionSide::call>(
#elif ROUGH_GAUSSIAN_TEST_MODEL == 2
        rb::launch_rough_stein_stein_european_option_cuda<OptionSide::call>(
#else
        rb::launch_log_modulated_rough_bergomi_european_option_cuda<OptionSide::call>(
#endif
            device_models.value, 1U,
            device_products.value, 2U,
            PriceConstruction::CartesianProduct, 2U, row_index,
            paths, 1.0f / 252.0f, 1.0f / 504.0f, 504U, chunk,
            reference_workspace.value, reference_bytes, seed,
            reference_prices.value, reference_errors.value
        );
    }
    check_cuda(cudaDeviceSynchronize(), "Gaussian price reference sync");
    float expected[2]{};
    reference_prices.download(expected, 2U);
    for (std::size_t i = 0U; i < 2U; ++i)
        require(std::abs(expected[i] - graph_prices[i]) < 5.0e-5f,
                "Gaussian graph central price differs from FFT pricer.");

    const auto delta_bytes =
        volterra::required_hybrid_fft_price_delta_workspace_bytes(
            504U, paths, chunk
        );
    DeviceBuffer<unsigned char> delta_workspace(delta_bytes);
    DeviceBuffer<float> delta_prices(2U), delta_price_errors(2U),
        deltas(2U), delta_errors(2U);
    for (std::size_t row_index = 0U; row_index < 2U; ++row_index) {
#if ROUGH_GAUSSIAN_TEST_MODEL == 0
        rb::launch_rough_bergomi_european_option_price_delta_cuda<
            OptionSide::call>(
#elif ROUGH_GAUSSIAN_TEST_MODEL == 1
        rb::launch_rough_sabr_european_option_price_delta_cuda<
            OptionSide::call>(
#elif ROUGH_GAUSSIAN_TEST_MODEL == 2
        rb::launch_rough_stein_stein_european_option_price_delta_cuda<
            OptionSide::call>(
#else
        rb::launch_log_modulated_rough_bergomi_european_option_price_delta_cuda<
            OptionSide::call>(
#endif
            models.data(), device_models.value, 1U,
            products.data(), device_products.value, 2U,
            PriceConstruction::CartesianProduct, 2U, row_index,
            paths, 1.0f / 252.0f, 1.0f / 504.0f, 504U, chunk,
            delta_workspace.value, delta_bytes, seed,
            {.02f}, delta_prices.value, delta_price_errors.value,
            deltas.value, delta_errors.value
        );
    }
    check_cuda(cudaDeviceSynchronize(), "Gaussian delta reference sync");
    float expected_delta[2]{};
    deltas.download(expected_delta, 2U);
    for (std::size_t row_index = 0U; row_index < 2U; ++row_index)
        require(std::abs(expected_delta[row_index]
                         - graph_gradients[row_index * 4U])
                    < 3.0e-3f,
                "Gaussian graph spot delta differs from the paired oracle.");
}
