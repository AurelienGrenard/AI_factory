// RH and QRH graph: CRN, Cartesian reuse, first/diagonal orders and price parity.
#include "common/check_cuda.cuh"
#include "model/equity/rough/rough_heston/product/european_option.cuh"
#include "model/equity/rough/rough_heston/product/european_option_price_gradients.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/european_option.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/european_option_price_gradients.cuh"

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
namespace rh = model::equity::rough_heston;
namespace qrh = model::equity::quadratic_rough_heston;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename T>
struct DeviceBuffer {
    T* value = nullptr;
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(&value, count * sizeof(T)), "rough graph allocate");
    }
    ~DeviceBuffer() { if (value) cudaFree(value); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void upload(const T* source, std::size_t count) {
        check_cuda(cudaMemcpy(value, source, count * sizeof(T),
                              cudaMemcpyHostToDevice), "rough graph upload");
    }
    void download(T* target, std::size_t count) const {
        check_cuda(cudaMemcpy(target, value, count * sizeof(T),
                              cudaMemcpyDeviceToHost), "rough graph download");
    }
};

template<typename Model, typename Prepared, typename Plan, typename Cache,
         template<OptionSide, std::size_t, pg::SensitivityOrders, std::size_t>
         class Graph,
         typename Prepare, typename Reference>
void exercise_model(
    const Model& model, Prepare prepare, Reference reference
) {
    constexpr float dt = 1.0f / 504.0f;
    constexpr std::size_t paths = 513U;
    constexpr std::uint64_t seed = 932000001U;
    const std::vector<Model> models{model};
    const std::vector<product::EuropeanOptionParameters> products{
        {1.0f, 252U}, {1.1f, 252U}
    };
    const pg::PriceGradientConfiguration configuration{{
        {"model.spot", {.01f, pg::BumpScale::relative}},
        {"model.hurst_exponent", {.001f, pg::BumpScale::absolute}},
        {"product.strike", {.01f, pg::BumpScale::relative}},
        {"product.maturity_years", {dt, pg::BumpScale::absolute}}
    }};
    const pg::LaunchConfiguration launch{
        pg::PricingMethod::monte_carlo, 0U, 2U, paths, 256U, 2U, seed
    };

    auto run = [&]<pg::SensitivityOrders Orders>() {
        Plan host = prepare(models, products, PriceConstruction::CartesianProduct,
                            pg::TimeConfiguration{dt, 2U}, configuration,
                            pg::SensitivityRequest{Orders});
        Cache cache(1.0f, dt);
        using Engine = Graph<OptionSide::call, 7U, Orders, 4U>;
        const auto execution = Engine::plan(host, cache, launch);
        require(cache.values().size() == 3U,
                "Cartesian spot/strike nodes must reuse prepared dynamics.");
        require(cache.unique_kernel_count() == 3U,
                "The H endpoints must each fit one distinct kernel.");
        const auto row = volterra::price_gradients::
            prepare_lift_graph_row<Orders, 4U>(host, 1U, cache);
        require(row.prepared_indices[0U] == row.prepared_indices[1U],
                "Spot endpoint duplicated the central dynamics.");
        require(row.prepared_indices[0U] == row.prepared_indices[5U],
                "Strike endpoint duplicated the central dynamics.");

        DeviceBuffer<unsigned char> workspace(execution.bytes);
        DeviceBuffer<float> prices(2U), price_errors(2U), gradients(8U),
            gradient_errors(8U), hessians(8U), hessian_errors(8U);
        DeviceBuffer<pg::SensitivityStencil<4U>> stencils(8U);
        pg::SensitivityOutputs outputs{
            prices.value, price_errors.value,
            gradients.value, gradient_errors.value,
            hessians.value, hessian_errors.value, 2U, 8U
        };
        mcpg::DevicePreparedStencilOutputs<4U> stencil_outputs{
            stencils.value, 8U, nullptr
        };
        Engine::launch(host, cache, launch, outputs, stencil_outputs,
                       execution, workspace.value, execution.bytes);
        check_cuda(cudaDeviceSynchronize(), "rough graph sync");
        float result[2]{}, error[2]{}, gradient[8]{}, hessian[8]{};
        prices.download(result, 2U);
        price_errors.download(error, 2U);
        gradients.download(gradient, 8U);
        if constexpr (pg::requests_second_v<Orders>)
            hessians.download(hessian, 8U);
        for (std::size_t row_index = 0U; row_index < 2U; ++row_index) {
            require(std::isfinite(result[row_index])
                    && std::isfinite(error[row_index])
                    && error[row_index] > 0.0f,
                    "Invalid rough graph price moments.");
            for (std::size_t sensitivity = 0U; sensitivity < 4U;
                 ++sensitivity) {
                const auto index = row_index * 4U + sensitivity;
                require(std::isfinite(gradient[index]),
                        "Invalid rough graph gradient.");
                if constexpr (pg::requests_second_v<Orders>)
                    require(std::isfinite(hessian[index]),
                            "Invalid rough graph diagonal Hessian.");
            }
        }
        return std::pair{std::vector<float>(result, result + 2U),
                         std::vector<float>(gradient, gradient + 8U)};
    };

    const auto both = run.template operator()<
        pg::SensitivityOrders::first_and_second>();
    const auto first = run.template operator()<pg::SensitivityOrders::first>();
    for (std::size_t i = 0U; i < 2U; ++i)
        require(both.first[i] == first.first[i],
                "Adding diagonal moments changed rough central prices.");
    for (std::size_t i = 0U; i < 8U; ++i)
        require(both.second[i] == first.second[i],
                "Adding diagonal moments changed rough gradients.");

    DeviceBuffer<Model> device_model(1U);
    DeviceBuffer<Prepared> device_prepared(1U);
    DeviceBuffer<product::EuropeanOptionParameters> device_products(2U);
    DeviceBuffer<float> reference_prices(2U), reference_errors(2U);
    Cache central(1.0f, dt);
    const auto central_index = central.index(model);
    device_model.upload(&model, 1U);
    device_prepared.upload(&central.values()[central_index], 1U);
    device_products.upload(products.data(), 2U);
    reference(device_model.value, device_prepared.value, products.data(),
              device_products.value, seed, paths,
              reference_prices.value, reference_errors.value);
    check_cuda(cudaDeviceSynchronize(), "rough reference sync");
    float expected[2]{};
    reference_prices.download(expected, 2U);
    for (std::size_t i = 0U; i < 2U; ++i)
        require(std::abs(expected[i] - both.first[i]) <= 1.0e-7f,
                "Graph central price differs from the existing lift pricer.");

    float endpoint_prices[2][2]{};
    for (std::size_t endpoint = 0U; endpoint < 2U; ++endpoint) {
        auto bumped = model;
        bumped.spot = model.spot * (endpoint == 0U ? .99f : 1.01f);
        const auto prepared_index = central.index(bumped);
        device_model.upload(&bumped, 1U);
        device_prepared.upload(
            &central.values()[prepared_index], 1U
        );
        reference(device_model.value, device_prepared.value,
                  products.data(), device_products.value,
                  seed, paths, reference_prices.value,
                  reference_errors.value);
        check_cuda(cudaDeviceSynchronize(), "rough bump reference sync");
        reference_prices.download(endpoint_prices[endpoint], 2U);
    }
    const float width = model.spot * 1.01f - model.spot * .99f;
    for (std::size_t row = 0U; row < 2U; ++row) {
        const float independent_delta =
            (endpoint_prices[1U][row] - endpoint_prices[0U][row])
            / width;
        require(std::abs(independent_delta - both.second[row * 4U])
                    < 3.0e-3f,
                "Rough graph spot delta differs from independent CRN prices.");
    }
}

template<typename Model>
void run_model(const Model& model) {
    if constexpr (std::is_same_v<Model, rh::ModelParameters>) {
        auto prepare = [](auto models, auto products, auto construction,
                          auto time, auto configuration, auto request) {
            return rh::prepare_rough_heston_european_option_sensitivities(
                models, products, construction, time, configuration, request);
        };
        auto reference = [](auto dm, auto dc, auto hp, auto dp, auto seed,
                            auto paths, auto prices, auto errors) {
            rh::launch_rough_heston_european_option_cuda<
                OptionSide::call, 7U>(
                dm, 1U, dc, 1U, hp, dp, 2U,
                PriceConstruction::CartesianProduct, 2U, 0U, 2U,
                paths, 1.0f / 504.0f, 2U, 256U, 2U, seed,
                prices, errors);
        };
        exercise_model<Model, rh::PreparedDynamics<7U>,
            rh::EuropeanOptionPriceGradientPlan,
            rh::price_gradients::PreparedLiftNodeCache<7U>,
            rh::EuropeanOptionNodeGraph>(model, prepare, reference);
    } else {
        auto prepare = [](auto models, auto products, auto construction,
                          auto time, auto configuration, auto request) {
            return qrh::prepare_quadratic_rough_heston_european_option_sensitivities(
                models, products, construction, time, configuration, request);
        };
        auto reference = [](auto dm, auto dc, auto hp, auto dp, auto seed,
                            auto paths, auto prices, auto errors) {
            qrh::launch_quadratic_rough_heston_european_option_cuda<
                OptionSide::call, 7U>(
                dm, 1U, dc, 1U, hp, dp, 2U,
                PriceConstruction::CartesianProduct, 2U, 0U, 2U,
                paths, 1.0f / 504.0f, 2U, 256U, 2U, seed,
                prices, errors);
        };
        exercise_model<Model, qrh::PreparedDynamics<7U>,
            qrh::EuropeanOptionPriceGradientPlan,
            qrh::price_gradients::PreparedLiftNodeCache<7U>,
            qrh::EuropeanOptionNodeGraph>(model, prepare, reference);
    }
}
} // namespace

int main() {
    int count = 0;
    const auto availability = cudaGetDeviceCount(&count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver || count == 0)
        return 77;
    check_cuda(availability, "rough graph cudaGetDeviceCount");
    run_model(rh::ModelParameters{
        1.0f, .02f, .01f, .04f, .30f, .02f, .30f, .10f, -.70f
    });
    run_model(qrh::ModelParameters{
        1.0f, .02f, .01f, .15f, .60f, .08f, .005f, 1.50f, 1.00f, .105371f
    });
}
