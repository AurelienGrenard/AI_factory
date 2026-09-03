// Exercise one- and two-factor Bermudan swaptions through the shared LSM engine.
#include "common/check_cuda.cuh"
#include "model/fixed_income/g2/product/bermudan_swaption.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/product/bermudan_swaption.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

namespace lsm = ai_factory::workbench::longstaff_schwartz;
namespace product = ai_factory::workbench::product;
using ai_factory::workbench::SwaptionSide;

constexpr std::size_t kRowCount = 2U;
constexpr std::size_t kPathsPerPrice = 4'096U;
constexpr unsigned int kThreadsPerBlock = 128U;
constexpr std::size_t kBlocksPerPrice = 16U;
constexpr float kDayFraction = 1.0f / 252.0f;
constexpr std::uint64_t kSeed = 2'170'000'001ULL;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct OrnsteinUhlenbeckAdapter {
    using ModelParameters = ai_factory::workbench::model::fixed_income::
        ornstein_uhlenbeck::ModelParameters;

    template<SwaptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::BermudanSwaptionParameters* host_products,
        const product::BermudanSwaptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::fixed_income::
            ornstein_uhlenbeck::
                launch_ornstein_uhlenbeck_bermudan_swaption_cuda<Side>(
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

struct G2Adapter {
    using ModelParameters =
        ai_factory::workbench::model::fixed_income::g2::ModelParameters;

    template<SwaptionSide Side>
    static lsm::LaunchResult launch(
        const ModelParameters* device_models,
        const product::BermudanSwaptionParameters* host_products,
        const product::BermudanSwaptionParameters* device_products,
        float* device_prices,
        float* device_standard_errors
    ) {
        return ai_factory::workbench::model::fixed_income::g2::
            launch_g2_bermudan_swaption_cuda<Side>(
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
    product::BermudanSwaptionParameters* products = nullptr;
    float* prices = nullptr;
    float* standard_errors = nullptr;

    ~DeviceArrays() {
        if (models != nullptr) cudaFree(models);
        if (products != nullptr) cudaFree(products);
        if (prices != nullptr) cudaFree(prices);
        if (standard_errors != nullptr) cudaFree(standard_errors);
    }
};

template<typename Adapter, SwaptionSide Side>
void validate_side(
    const DeviceArrays<Adapter>& device,
    const std::array<product::BermudanSwaptionParameters, kRowCount>& products
) {
    std::array<float, kRowCount> first_prices{};
    std::array<float, kRowCount> first_errors{};
    std::array<float, kRowCount> second_prices{};
    std::array<float, kRowCount> second_errors{};

    const auto price_once = [&] (
        std::array<float, kRowCount>& prices,
        std::array<float, kRowCount>& errors
    ) {
        const lsm::LaunchResult result = Adapter::template launch<Side>(
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
                sizeof(prices),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy Bermudan prices"
        );
        ai_factory::workbench::check_cuda(
            cudaMemcpy(
                errors.data(),
                device.standard_errors,
                sizeof(errors),
                cudaMemcpyDeviceToHost
            ),
            "cudaMemcpy Bermudan standard errors"
        );
        return result;
    };

    const lsm::LaunchResult first = price_once(first_prices, first_errors);
    const lsm::LaunchResult second = price_once(second_prices, second_errors);
    lsm::validate_regression_diagnostics(first, "Bermudan swaption");
    lsm::validate_regression_diagnostics(second, "Bermudan swaption");
    require(first.kernel_seconds > 0.0, "No Bermudan kernel time was recorded.");
    require(first.kernel_launch_count > 0U, "No Bermudan kernel was launched.");
    require(
        first.regression_diagnostics.successful_regression_count > 0U,
        "No Bermudan regression was solved."
    );
    for (std::size_t row = 0U; row < kRowCount; ++row) {
        require(
            std::isfinite(first_prices[row]) && first_prices[row] >= 0.0f,
            "A Bermudan swaption price is invalid."
        );
        require(
            std::isfinite(first_errors[row]) && first_errors[row] >= 0.0f,
            "A Bermudan swaption standard error is invalid."
        );
        require(
            first_prices[row] == second_prices[row]
                && first_errors[row] == second_errors[row],
            "A Bermudan swaption result is not bitwise reproducible."
        );
    }
}

template<typename Adapter>
void validate_model(
    const std::array<typename Adapter::ModelParameters, kRowCount>& models,
    const std::array<product::BermudanSwaptionParameters, kRowCount>& products
) {
    DeviceArrays<Adapter> device;
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.models, sizeof(models)),
        "cudaMalloc Bermudan models"
    );
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.products, sizeof(products)),
        "cudaMalloc Bermudan products"
    );
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.prices, kRowCount * sizeof(float)),
        "cudaMalloc Bermudan prices"
    );
    ai_factory::workbench::check_cuda(
        cudaMalloc(&device.standard_errors, kRowCount * sizeof(float)),
        "cudaMalloc Bermudan standard errors"
    );
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            device.models,
            models.data(),
            sizeof(models),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy Bermudan models"
    );
    ai_factory::workbench::check_cuda(
        cudaMemcpy(
            device.products,
            products.data(),
            sizeof(products),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy Bermudan products"
    );

    validate_side<Adapter, SwaptionSide::payer>(device, products);
    validate_side<Adapter, SwaptionSide::receiver>(device, products);
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

    const std::array<product::BermudanSwaptionParameters, kRowCount> products{{
        {1.0f, 0.005f, 1.0f, 126U, 252U, 2U, 2U},
        {1.0f, 0.040f, 0.5f, 252U, 126U, 8U, 4U},
    }};
    const std::array<OrnsteinUhlenbeckAdapter::ModelParameters, kRowCount>
        ou_models{{
            {{0.3572258f, 0.0180200f}, 0.0384505f},
            {{0.7992920f, 0.0270289f}, 0.0368955f},
        }};
    const std::array<G2Adapter::ModelParameters, kRowCount> g2_models{{
        {{0.1910284f, 0.0016900f, 0.4747388f, 0.0104169f, 0.2112323f},
         {0.0489586f, 0.0170112f}},
        {{0.2027372f, 0.0035430f, 1.0933268f, 0.0146284f, 0.1545814f},
         {0.0023029f, 0.0024545f}},
    }};

    validate_model<OrnsteinUhlenbeckAdapter>(ou_models, products);
    validate_model<G2Adapter>(g2_models, products);
}
