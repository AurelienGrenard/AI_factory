// Non-European terminal product exercises the shared rough graph payoff policy.
#include "common/check_cuda.cuh"
#if ROUGH_TERMINAL_TEST_MODEL == 0
#include "model/equity/rough/rough_heston/product/digital_option.cuh"
#include "model/equity/rough/rough_heston/product/digital_option_price_gradients.cuh"
#else
#include "model/equity/rough/rough_bergomi/product/digital_option.cuh"
#include "model/equity/rough/rough_bergomi/product/digital_option_price_gradients.cuh"
#endif

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace mcpg = monte_carlo::price_gradients;
#if ROUGH_TERMINAL_TEST_MODEL == 0
namespace rough_model = model::equity::rough_heston;
#else
namespace rough_model = model::equity::rough_bergomi;
#endif
template<typename T>
struct DeviceBuffer {
    T* pointer = nullptr;
    explicit DeviceBuffer(std::size_t n) {
        check_cuda(cudaMalloc(&pointer, n * sizeof(T)), "rough digital malloc");
    }
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* p, std::size_t n) {
        check_cuda(cudaMemcpy(pointer, p, n * sizeof(T),
                              cudaMemcpyHostToDevice), "rough digital upload");
    }
    void download(T* p, std::size_t n) {
        check_cuda(cudaMemcpy(p, pointer, n * sizeof(T),
                              cudaMemcpyDeviceToHost), "rough digital download");
    }
};
void require(bool ok) {
    if (!ok) throw std::runtime_error("Rough digital graph parity failed.");
}
}

int main() {
    int count = 0;
    const auto availability = cudaGetDeviceCount(&count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver || count == 0)
        return 77;
    check_cuda(availability, "rough digital cudaGetDeviceCount");
#if ROUGH_TERMINAL_TEST_MODEL == 0
    const std::vector<rough_model::ModelParameters> models{{
        1.0f, .02f, .01f, .04f, .30f, .02f, .30f, .10f, -.70f
    }};
#else
    const std::vector<rough_model::ModelParameters> models{{
        1.0f, .02f, .01f, .04f, .8f, .10f, -.7f
    }};
#endif
    const std::vector<product::DigitalOptionParameters> products{
        {1.0f, 252U, 1.0f}
    };
    constexpr float dt = 1.0f / 504.0f;
    constexpr std::size_t paths = 513U;
    constexpr std::uint64_t seed = 932000001U;
    const pg::PriceGradientConfiguration selection{{
        {"model.spot", {.01f, pg::BumpScale::relative}},
        {"product.strike", {.01f, pg::BumpScale::relative}},
        {"product.maturity_years", {dt, pg::BumpScale::absolute}}
    }};
#if ROUGH_TERMINAL_TEST_MODEL == 0
    const auto host = rough_model::prepare_rough_heston_digital_option_sensitivities(
#else
    const auto host = rough_model::prepare_rough_bergomi_digital_option_sensitivities(
#endif
        models, products, PriceConstruction::Aligned, {dt, 2U},
        selection, {pg::SensitivityOrders::first_and_second}
    );
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, 1U,
        paths, 256U, 1U, seed
    };
#if ROUGH_TERMINAL_TEST_MODEL == 0
    using Graph = rough_model::DigitalOptionNodeGraph<
        OptionSide::call, 7U, pg::SensitivityOrders::first_and_second, 3U
    >;
    rough_model::price_gradients::PreparedLiftNodeCache<7U> cache(1.0f, dt);
#else
    using Graph = rough_model::DigitalOptionNodeGraph<
        OptionSide::call, pg::SensitivityOrders::first_and_second, 3U
    >;
    Graph::Cache cache;
#endif
    const auto execution = Graph::plan(host, cache, launch);
    DeviceBuffer<unsigned char> workspace(execution.bytes);
    DeviceBuffer<float> prices(1U), errors(1U), gradients(3U),
        gradient_errors(3U), hessians(3U), hessian_errors(3U);
    DeviceBuffer<pg::SensitivityStencil<4U>> stencils(3U);
    pg::SensitivityOutputs outputs{
        prices.pointer, errors.pointer,
        gradients.pointer, gradient_errors.pointer,
        hessians.pointer, hessian_errors.pointer, 1U, 3U
    };
    mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs{
        stencils.pointer, 3U, nullptr
    };
    Graph::launch(host, cache, launch, outputs, stencil_outputs,
                  execution, workspace.pointer, execution.bytes);
    check_cuda(cudaDeviceSynchronize(), "rough digital graph sync");
    float graph_price = 0.0f, graph_errors = 0.0f;
    float gradient[3]{}, hessian[3]{};
    prices.download(&graph_price, 1U);
    errors.download(&graph_errors, 1U);
    gradients.download(gradient, 3U);
    hessians.download(hessian, 3U);
    require(std::isfinite(graph_price) && graph_price > 0.0f
            && std::isfinite(graph_errors) && graph_errors > 0.0f);
    for (std::size_t i = 0U; i < 3U; ++i)
        require(std::isfinite(gradient[i]) && std::isfinite(hessian[i]));

    DeviceBuffer<rough_model::ModelParameters> device_model(1U);
    DeviceBuffer<product::DigitalOptionParameters> device_product(1U);
    DeviceBuffer<float> reference_price(1U), reference_error(1U);
    device_model.upload(models.data(), 1U);
    device_product.upload(products.data(), 1U);
#if ROUGH_TERMINAL_TEST_MODEL == 0
    DeviceBuffer<rough_model::PreparedDynamics<7U>> prepared(1U);
    prepared.upload(&cache.values()[0U], 1U);
    rough_model::launch_rough_heston_digital_option_cuda<OptionSide::call, 7U>(
        device_model.pointer, 1U, prepared.pointer, 1U,
        products.data(), device_product.pointer, 1U,
        PriceConstruction::Aligned, 1U, 0U, 1U,
        paths, dt, 2U, 256U, 1U, seed,
        reference_price.pointer, reference_error.pointer
    );
#else
    const auto bytes =
        volterra::required_hybrid_fft_workspace_bytes(504U, paths, 256U);
    DeviceBuffer<unsigned char> reference_workspace(bytes);
    rough_model::launch_rough_bergomi_digital_option_cuda<OptionSide::call>(
        device_model.pointer, 1U, device_product.pointer, 1U,
        PriceConstruction::Aligned, 1U, 0U, paths,
        1.0f / 252.0f, dt, 504U, 256U,
        reference_workspace.pointer, bytes, seed,
        reference_price.pointer, reference_error.pointer
    );
#endif
    check_cuda(cudaDeviceSynchronize(), "rough digital reference sync");
    float expected = 0.0f;
    reference_price.download(&expected, 1U);
    require(std::abs(expected - graph_price) < 5.0e-5f);
}
