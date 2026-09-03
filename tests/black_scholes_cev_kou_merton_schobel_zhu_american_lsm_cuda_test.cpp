// Exercise American LSM composition for Black-Scholes, CEV, Kou, Merton and Schobel-Zhu.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/black_scholes/product/american_option.cuh"
#include "model/equity/markovian/cev/product/american_option.cuh"
#include "model/equity/markovian/kou/product/american_option.cuh"
#include "model/equity/markovian/merton/product/american_option.cuh"
#include "model/equity/markovian/schobel_zhu/product/american_option.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

using ai_factory::workbench::OptionSide;
namespace lsm = ai_factory::workbench::longstaff_schwartz;
namespace product = ai_factory::workbench::product;

constexpr std::size_t kRowCount = 2U;
constexpr std::size_t kPathsPerPrice = 2'048U;
constexpr unsigned int kThreadsPerBlock = 256U;
constexpr std::size_t kBlocksPerPrice = 8U;
constexpr float kDayFraction = 1.0f / 252.0f;
constexpr float kDt = 1.0f / 252.0f;
constexpr std::uint32_t kSimulationStepsPerDay = 1U;
constexpr std::uint64_t kSeed = 900000001ULL;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct BlackScholesAdapter {
    using ModelParameters =
        ai_factory::workbench::model::equity::black_scholes::ModelParameters;

    template<OptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::AmericanOptionParameters* host_products,
        const product::AmericanOptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::equity::black_scholes::
            launch_black_scholes_american_option_cuda<Side>(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                ai_factory::workbench::PriceConstruction::Aligned, kRowCount, kPathsPerPrice, kDayFraction,
                kThreadsPerBlock, kBlocksPerPrice, kSeed,
                device_prices, device_standard_errors
            );
    }

};

struct CevAdapter {
    using ModelParameters =
        ai_factory::workbench::model::equity::cev::ModelParameters;

    template<OptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::AmericanOptionParameters* host_products,
        const product::AmericanOptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::equity::cev::
            launch_cev_american_option_cuda<Side>(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                ai_factory::workbench::PriceConstruction::Aligned, kRowCount, kPathsPerPrice, kDt,
                kSimulationStepsPerDay,
                kThreadsPerBlock, kBlocksPerPrice, kSeed,
                device_prices, device_standard_errors
            );
    }
};

struct KouAdapter {
    using ModelParameters =
        ai_factory::workbench::model::equity::kou::ModelParameters;

    template<OptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::AmericanOptionParameters* host_products,
        const product::AmericanOptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::equity::kou::
            launch_kou_american_option_cuda<Side>(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                ai_factory::workbench::PriceConstruction::Aligned, kRowCount, kPathsPerPrice, kDayFraction,
                kThreadsPerBlock, kBlocksPerPrice, kSeed,
                device_prices, device_standard_errors
            );
    }
};

struct MertonAdapter {
    using ModelParameters =
        ai_factory::workbench::model::equity::merton::ModelParameters;

    template<OptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::AmericanOptionParameters* host_products,
        const product::AmericanOptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::equity::merton::
            launch_merton_american_option_cuda<Side>(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                ai_factory::workbench::PriceConstruction::Aligned, kRowCount, kPathsPerPrice, kDayFraction,
                kThreadsPerBlock, kBlocksPerPrice, kSeed,
                device_prices, device_standard_errors
            );
    }
};

struct SchobelZhuAdapter {
    using ModelParameters =
        ai_factory::workbench::model::equity::schobel_zhu::ModelParameters;

    template<OptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::AmericanOptionParameters* host_products,
        const product::AmericanOptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::equity::schobel_zhu::
            launch_schobel_zhu_american_option_cuda<Side>(
                device_models, kRowCount,
                host_products, device_products, kRowCount,
                ai_factory::workbench::PriceConstruction::Aligned, kRowCount, kPathsPerPrice, kDt,
                kSimulationStepsPerDay,
                kThreadsPerBlock, kBlocksPerPrice, kSeed,
                device_prices, device_standard_errors
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
    std::vector<float> prices(kRowCount);
    std::vector<float> standard_errors(kRowCount);
    const lsm::LaunchResult execution = Adapter::template launch<Side>(
        device.models,
        products.data(),
        device.products,
        device.prices,
        device.standard_errors
    );
    lsm::validate_regression_diagnostics(
        execution, "Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM"
    );
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            prices.data(), device.prices,
            kRowCount * sizeof(float), cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM prices"
    );
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            standard_errors.data(), device.standard_errors,
            kRowCount * sizeof(float), cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM errors"
    );

    require(execution.batch_count >= 1U, "No LSM batch was launched.");
    require(
        execution.kernel_launch_count >= 1U,
        "No LSM kernel was launched."
    );
    require(
        execution.blocks_per_price == kBlocksPerPrice,
        "Unexpected LSM block count."
    );
    require(execution.workspace_bytes > 0U, "Empty LSM workspace.");

    for (std::size_t row = 0U; row < kRowCount; ++row) {
        const float intrinsic = Side == OptionSide::call
            ? std::max(models[row].spot - products[row].strike, 0.0f)
            : std::max(products[row].strike - models[row].spot, 0.0f);
        require(std::isfinite(prices[row]), "Non-finite American price.");
        require(
            std::isfinite(standard_errors[row]),
            "Non-finite American standard error."
        );
        require(
            standard_errors[row] >= 0.0f,
            "Negative American standard error."
        );
        require(
            prices[row] + 1.0e-6f >= intrinsic,
            "American price is below immediate exercise."
        );
    }
}

template<typename Adapter>
void validate_model(
    const std::vector<typename Adapter::ModelParameters>& models,
    const std::vector<product::AmericanOptionParameters>& products
) {
    DeviceArrays<Adapter> device;
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.models, kRowCount * sizeof(models.front())),
        "cudaMalloc Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM models"
    );
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.products, kRowCount * sizeof(products.front())),
        "cudaMalloc Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM products"
    );
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.prices, kRowCount * sizeof(float)),
        "cudaMalloc Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM prices"
    );
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.standard_errors, kRowCount * sizeof(float)),
        "cudaMalloc Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM errors"
    );
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            device.models, models.data(),
            kRowCount * sizeof(models.front()), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM models"
    );
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            device.products, products.data(),
            kRowCount * sizeof(products.front()), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy Black-Scholes/CEV/Kou/Merton/Schobel-Zhu American LSM products"
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
        {1.10f, 252U, 21U},
    };
    validate_model<BlackScholesAdapter>(
        {
            {1.0f, 0.02f, 0.01f, 0.20f},
            {1.0f, 0.04f, 0.02f, 0.30f},
        },
        products
    );
    validate_model<CevAdapter>(
        {
            {1.0f, 0.02f, 0.01f, 0.20f, 0.75f},
            {1.0f, 0.04f, 0.02f, 0.30f, 0.60f},
        },
        products
    );
    validate_model<KouAdapter>(
        {
            {1.0f, 0.02f, 0.01f, 0.20f, 0.40f, 0.40f, 10.0f, 5.0f},
            {1.0f, 0.04f, 0.02f, 0.30f, 0.80f, 0.35f, 8.0f, 4.0f},
        },
        products
    );
    validate_model<MertonAdapter>(
        {
            {1.0f, 0.02f, 0.01f, 0.20f, 0.30f, -0.10f, 0.20f},
            {1.0f, 0.04f, 0.02f, 0.30f, 0.60f, -0.15f, 0.25f},
        },
        products
    );
    validate_model<SchobelZhuAdapter>(
        {
            {1.0f, 0.02f, 0.01f, 0.20f, 2.0f, 0.20f, 0.25f, -0.60f},
            {1.0f, 0.04f, 0.02f, 0.30f, 1.5f, 0.25f, 0.35f, -0.40f},
        },
        products
    );
}
