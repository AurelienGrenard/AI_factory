// Exercise arithmetic and geometric Asian products on the QRH N-factor engine.
#include "common/check_cuda.cuh"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/quadratic_rough_heston/product/asian_option.cuh"
#include "model/equity/rough/quadratic_rough_heston/product/geometric_asian_option.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

using namespace ai_factory::workbench;
namespace quadratic = model::equity::quadratic_rough_heston;

struct PriceResult {
    float price;
    float standard_error;
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename ProductParameters, typename Launch>
PriceResult run_product(
    const quadratic::ModelParameters& model,
    const quadratic::PreparedDynamics<7U>& prepared,
    const ProductParameters& product,
    Launch&& launch
) {
    quadratic::ModelParameters* device_model = nullptr;
    quadratic::PreparedDynamics<7U>* device_prepared = nullptr;
    ProductParameters* device_product = nullptr;
    float* device_price = nullptr;
    float* device_standard_error = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "QRH Asian model");
    check_cuda(
        cudaMalloc(&device_prepared, sizeof(prepared)),
        "QRH Asian prepared dynamics"
    );
    check_cuda(
        cudaMalloc(&device_product, sizeof(product)),
        "QRH Asian product"
    );
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "QRH Asian price");
    check_cuda(
        cudaMalloc(&device_standard_error, sizeof(float)),
        "QRH Asian standard error"
    );
    check_cuda(
        cudaMemcpy(device_model, &model, sizeof(model), cudaMemcpyHostToDevice),
        "QRH Asian model copy"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            &prepared,
            sizeof(prepared),
            cudaMemcpyHostToDevice
        ),
        "QRH Asian prepared copy"
    );
    check_cuda(
        cudaMemcpy(
            device_product,
            &product,
            sizeof(product),
            cudaMemcpyHostToDevice
        ),
        "QRH Asian product copy"
    );

    auto execute = [&] {
        launch(
            device_model,
            device_prepared,
            &product,
            device_product,
            device_price,
            device_standard_error
        );
        check_cuda(cudaDeviceSynchronize(), "QRH Asian synchronize");
        PriceResult result{};
        check_cuda(
            cudaMemcpy(
                &result.price,
                device_price,
                sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "QRH Asian price copy"
        );
        check_cuda(
            cudaMemcpy(
                &result.standard_error,
                device_standard_error,
                sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "QRH Asian standard-error copy"
        );
        return result;
    };
    const PriceResult first = execute();
    const PriceResult replay = execute();
    require(
        first.price == replay.price
            && first.standard_error == replay.standard_error,
        "QRH Asian product did not replay bitwise."
    );
    require(
        std::isfinite(first.price) && first.price >= 0.0f
            && std::isfinite(first.standard_error)
            && first.standard_error >= 0.0f,
        "QRH Asian product returned invalid statistics."
    );

    check_cuda(cudaFree(device_standard_error), "free QRH Asian error");
    check_cuda(cudaFree(device_price), "free QRH Asian price");
    check_cuda(cudaFree(device_product), "free QRH Asian product");
    check_cuda(cudaFree(device_prepared), "free QRH Asian prepared dynamics");
    check_cuda(cudaFree(device_model), "free QRH Asian model");
    return first;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "QRH Asian cudaGetDeviceCount");

    const quadratic::ModelParameters model{
        1.0f, 0.02f, 0.01f, 0.15f, 0.60f,
        0.08f, 0.005f, 1.50f, 1.00f, 0.105371f,
    };
    constexpr float dt = 1.0f / 504.0f;
    constexpr std::size_t path_count = 32'768U;
    constexpr unsigned int threads_per_block = 256U;
    constexpr std::size_t block_count = 1U;
    constexpr std::uint64_t seed = 11668828425518317568ULL;
    const auto prepared = quadratic::prepare_dynamics<7U>(model, 1.0f, dt);

    const PriceResult arithmetic = run_product(
        model,
        prepared,
        product::AsianOptionParameters{1.0f, 252U},
        [&](const auto* device_model, const auto* device_prepared,
            const auto* host_product,
            const auto* device_product, float* device_price,
            float* device_standard_error) {
            quadratic::launch_quadratic_rough_heston_asian_option_cuda<
                OptionSide::call,
                7U
            >(
                device_model, 1U, device_prepared, 1U,
                host_product, device_product, 1U,
                PriceConstruction::Aligned, 1U, 0U, 1U,
                path_count, dt, 2U, threads_per_block, block_count, seed,
                device_price, device_standard_error
            );
        }
    );
    const PriceResult geometric = run_product(
        model,
        prepared,
        product::GeometricAsianOptionParameters{1.0f, 252U},
        [&](const auto* device_model, const auto* device_prepared,
            const auto* host_product,
            const auto* device_product, float* device_price,
            float* device_standard_error) {
            quadratic::launch_quadratic_rough_heston_geometric_asian_option_cuda<
                OptionSide::call,
                7U
            >(
                device_model, 1U, device_prepared, 1U,
                host_product, device_product, 1U,
                PriceConstruction::Aligned, 1U, 0U, 1U,
                path_count, dt, 2U, threads_per_block, block_count, seed,
                device_price, device_standard_error
            );
        }
    );
    require(
        geometric.price <= arithmetic.price
            + 6.0f * std::hypot(
                geometric.standard_error,
                arithmetic.standard_error
            ),
        "QRH geometric Asian call exceeded the arithmetic Asian call."
    );
    return 0;
}
