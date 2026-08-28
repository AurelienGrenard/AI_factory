// Exercise the shared early-exercise engine with both exact Levy schedules.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/product/american_option.cuh"
#include "model/equity/markovian/variance_gamma/product/american_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using ai_factory::workbench::OptionSide;
namespace product = ai_factory::workbench::product;
namespace lsm = ai_factory::workbench::longstaff_schwartz;

constexpr std::size_t kRowCount = 4U;
constexpr std::size_t kPathsPerPrice = 4'096U;
constexpr unsigned int kThreadsPerBlock = 256U;
constexpr std::size_t kBlocksPerPrice = 16U;
constexpr float kDayFraction = 1.0f / 252.0f;
constexpr std::uint64_t kSeed = 900000001ULL;
constexpr std::size_t kExpectedKernelLaunchCount = 154U;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct VarianceGammaAdapter {
    using ModelParameters =
        ai_factory::workbench::model::equity::variance_gamma::ModelParameters;

    template<OptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::AmericanOptionParameters* host_products,
        const product::AmericanOptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::equity::variance_gamma::
            launch_variance_gamma_american_option_cuda<Side>(
                device_models,
                kRowCount,
                host_products,
                device_products,
                kRowCount,
                ai_factory::workbench::PriceConstruction::Aligned,
                kRowCount,
                kPathsPerPrice,
                kDayFraction,
                kThreadsPerBlock,
                kBlocksPerPrice,
                kSeed,
                device_prices,
                device_standard_errors
            );
    }
};

struct NigAdapter {
    using ModelParameters =
        ai_factory::workbench::model::equity::normal_inverse_gaussian::
            ModelParameters;

    template<OptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::AmericanOptionParameters* host_products,
        const product::AmericanOptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::equity::normal_inverse_gaussian::
            launch_normal_inverse_gaussian_american_option_cuda<Side>(
                device_models,
                kRowCount,
                host_products,
                device_products,
                kRowCount,
                ai_factory::workbench::PriceConstruction::Aligned,
                kRowCount,
                kPathsPerPrice,
                kDayFraction,
                kThreadsPerBlock,
                kBlocksPerPrice,
                kSeed,
                device_prices,
                device_standard_errors
            );
    }
};

template<typename Adapter>
struct DeviceArrays {
    typename Adapter::ModelParameters* models = nullptr;
    product::AmericanOptionParameters* products = nullptr;
    float* prices = nullptr;
    float* standard_errors = nullptr;

    ~DeviceArrays() {
        if (models != nullptr) cudaFree(models);
        if (products != nullptr) cudaFree(products);
        if (prices != nullptr) cudaFree(prices);
        if (standard_errors != nullptr) cudaFree(standard_errors);
    }
};

template<typename Adapter, OptionSide Side>
void validate_side(
    const DeviceArrays<Adapter>& device,
    const std::vector<typename Adapter::ModelParameters>& models,
    const std::vector<product::AmericanOptionParameters>& products
) {
    std::vector<float> first_prices(kRowCount);
    std::vector<float> first_errors(kRowCount);
    std::vector<float> second_prices(kRowCount);
    std::vector<float> second_errors(kRowCount);

    const auto price_once = [&] (
        std::vector<float>& prices,
        std::vector<float>& errors
    ) {
        const lsm::LaunchResult execution = Adapter::template launch<Side>(
            device.models,
            products.data(),
            device.products,
            device.prices,
            device.standard_errors
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                prices.data(),
                device.prices,
                kRowCount * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy exact-Levy American prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                errors.data(),
                device.standard_errors,
                kRowCount * sizeof(float),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy exact-Levy American errors"
        );
        return execution;
    };

    const lsm::LaunchResult first = price_once(first_prices, first_errors);
    const lsm::LaunchResult second = price_once(second_prices, second_errors);
    lsm::validate_regression_diagnostics(first, "Levy American option");
    lsm::validate_regression_diagnostics(second, "Levy American option");
    require(
        first.kernel_launch_count == kExpectedKernelLaunchCount,
        "The backward induction launched an unexpected kernel count."
    );
    require(
        first.batch_count == second.batch_count
            && first.kernel_launch_count == second.kernel_launch_count
            && first.maximum_prices_per_batch
                == second.maximum_prices_per_batch
            && first.blocks_per_price == second.blocks_per_price
            && first.workspace_bytes == second.workspace_bytes,
        "The exact-Levy execution plan is not reproducible."
    );

    for (std::size_t row = 0U; row < kRowCount; ++row) {
        const float intrinsic = Side == OptionSide::call
            ? std::max(models[row].spot - products[row].strike, 0.0f)
            : std::max(products[row].strike - models[row].spot, 0.0f);
        require(std::isfinite(first_prices[row]), "Non-finite American price.");
        require(std::isfinite(first_errors[row]), "Non-finite standard error.");
        require(first_errors[row] >= 0.0f, "Negative standard error.");
        require(
            first_prices[row] + 1.0e-6f >= intrinsic,
            "American price is below immediate exercise."
        );
        require(
            first_prices[row] == second_prices[row]
                && first_errors[row] == second_errors[row],
            "The exact-Levy American output is not bitwise reproducible."
        );
    }
}

template<typename Adapter>
void validate_model(
    const std::vector<typename Adapter::ModelParameters>& models,
    const std::vector<product::AmericanOptionParameters>& products
) {
    using ai_factory::workbench::check_cuda;
    DeviceArrays<Adapter> device;
    check_cuda(
        cudaMalloc(&device.models, kRowCount * sizeof(models.front())),
        "cudaMalloc exact-Levy models"
    );
    check_cuda(
        cudaMalloc(&device.products, kRowCount * sizeof(products.front())),
        "cudaMalloc American products"
    );
    check_cuda(
        cudaMalloc(&device.prices, kRowCount * sizeof(float)),
        "cudaMalloc American prices"
    );
    check_cuda(
        cudaMalloc(&device.standard_errors, kRowCount * sizeof(float)),
        "cudaMalloc American errors"
    );
    check_cuda(
        cudaMemcpy(
            device.models,
            models.data(),
            kRowCount * sizeof(models.front()),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy exact-Levy models"
    );
    check_cuda(
        cudaMemcpy(
            device.products,
            products.data(),
            kRowCount * sizeof(products.front()),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy American products"
    );

    validate_side<Adapter, OptionSide::call>(device, models, products);
    validate_side<Adapter, OptionSide::put>(device, models, products);
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    ai_factory::workbench::check_cuda(
        availability, "test cudaGetDeviceCount"
    );

    const std::vector<product::AmericanOptionParameters> products = {
        {0.90f, 126U, 21U},
        {1.00f, 252U, 21U},
        {1.10f, 378U, 10U},
        {1.20f, 504U, 10U},
    };
    const std::vector<VarianceGammaAdapter::ModelParameters> vg_models = {
        {1.0f, 0.02f, 0.01f, 0.20f, 0.20f, -0.10f},
        {1.0f, 0.03f, 0.00f, 0.25f, 0.15f, -0.05f},
        {1.0f, 0.01f, 0.02f, 0.18f, 0.30f, 0.05f},
        {1.0f, 0.04f, 0.01f, 0.30f, 0.10f, -0.15f},
    };
    const std::vector<NigAdapter::ModelParameters> nig_models = {
        {1.0f, 0.02f, 0.01f, 8.0f, -2.0f, 0.20f},
        {1.0f, 0.03f, 0.00f, 10.0f, -1.5f, 0.25f},
        {1.0f, 0.01f, 0.02f, 7.0f, -2.5f, 0.15f},
        {1.0f, 0.04f, 0.01f, 12.0f, -3.0f, 0.30f},
    };

    validate_model<VarianceGammaAdapter>(vg_models, products);
    validate_model<NigAdapter>(nig_models, products);
}
