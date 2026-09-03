// Exercise Bates terminal-payoff launchers and payoff identities.
#include "common/check_cuda.cuh"
#include "model/equity/markovian/bates/product/asian_option.cuh"
#include "model/equity/markovian/bates/product/asset_or_nothing_option.cuh"
#include "model/equity/markovian/bates/product/digital_option.cuh"
#include "model/equity/markovian/bates/product/double_knock_out_option.cuh"
#include "model/equity/markovian/bates/product/down_and_in_option.cuh"
#include "model/equity/markovian/bates/product/down_and_out_option.cuh"
#include "model/equity/markovian/bates/product/european_option.cuh"
#include "model/equity/markovian/bates/product/gap_option.cuh"
#include "model/equity/markovian/bates/product/forward_start_option.cuh"
#include "model/equity/markovian/bates/product/geometric_asian_option.cuh"
#include "model/equity/markovian/bates/product/straddle.cuh"
#include "model/equity/markovian/bates/product/up_and_in_option.cuh"
#include "model/equity/markovian/bates/product/up_no_touch.cuh"
#include "model/equity/markovian/bates/product/up_one_touch.cuh"
#include "model/equity/markovian/bates/product/up_and_out_option.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr std::size_t kPathsPerPrice = 16'384U;
constexpr float kDt = 1.0f / 504.0f;
constexpr std::uint32_t kSimulationStepsPerDay = 2U;
constexpr unsigned int kThreadsPerBlock = 256U;
constexpr std::uint64_t kSeed = 900000001ULL;

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Bound differences between estimators that no longer share every random draw.
float six_sigma(float first, float second, float third = 0.0f) {
    return 6.0f * std::sqrt(
        first * first + second * second + third * third
    );
}

// Launch one product row and return its price and standard error.
template <typename Product, typename Launcher>
void price_one(
    const ai_factory::workbench::model::equity::bates::ModelParameters& model,
    const Product& product,
    Launcher launch,
    float& price,
    float& standard_error
) {
    using namespace ai_factory::workbench;

    ai_factory::workbench::model::equity::bates::ModelParameters* device_model = nullptr;
    Product* device_product = nullptr;
    float* device_price = nullptr;
    float* device_standard_error = nullptr;
    check_cuda(cudaMalloc(&device_model, sizeof(model)), "test cudaMalloc model");
    check_cuda(cudaMalloc(&device_product, sizeof(product)), "test cudaMalloc product");
    check_cuda(cudaMalloc(&device_price, sizeof(float)), "test cudaMalloc price");
    check_cuda(
        cudaMalloc(&device_standard_error, sizeof(float)),
        "test cudaMalloc standard error"
    );
    check_cuda(
        cudaMemcpy(device_model, &model, sizeof(model), cudaMemcpyHostToDevice),
        "test cudaMemcpy model"
    );
    check_cuda(
        cudaMemcpy(
            device_product, &product, sizeof(product), cudaMemcpyHostToDevice
        ),
        "test cudaMemcpy product"
    );

    launch(
        device_model, 1U, &product, device_product, 1U, ai_factory::workbench::PriceConstruction::Aligned, 1U, 0U, 1U,
        kPathsPerPrice, kDt, kSimulationStepsPerDay, kThreadsPerBlock, 1U, kSeed,
        device_price, device_standard_error
    );
    check_cuda(cudaDeviceSynchronize(), "test payoff kernel synchronize");
    check_cuda(
        cudaMemcpy(&price, device_price, sizeof(float), cudaMemcpyDeviceToHost),
        "test cudaMemcpy price"
    );
    check_cuda(
        cudaMemcpy(
            &standard_error,
            device_standard_error,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "test cudaMemcpy standard error"
    );

    check_cuda(cudaFree(device_model), "test cudaFree model");
    check_cuda(cudaFree(device_product), "test cudaFree product");
    check_cuda(cudaFree(device_price), "test cudaFree price");
    check_cuda(cudaFree(device_standard_error), "test cudaFree standard error");
}

}  // namespace

// Verify all launchers and identities with deterministic Monte Carlo streams.
int main() {
    using namespace ai_factory::workbench;

    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "terminal-payoff test cudaGetDeviceCount");

    const ai_factory::workbench::model::equity::bates::ModelParameters model = {
        1.0f, 0.02f, 0.01f, 0.04f, 1.5f, 0.04f, 0.30f, -0.70f,
        0.40f, -0.10f, 0.20f,
    };
    float call = 0.0f;
    float put = 0.0f;
    float error = 0.0f;
    float call_error = 0.0f;
    float put_error = 0.0f;
    price_one(
        model,
        product::EuropeanOptionParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_european_option_cuda<OptionSide::call>,
        call,
        call_error
    );
    price_one(
        model,
        product::EuropeanOptionParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_european_option_cuda<OptionSide::put>,
        put,
        put_error
    );

    float straddle = 0.0f;
    price_one(
        model,
        product::StraddleParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_straddle_cuda,
        straddle,
        error
    );
    require(
        std::fabs(straddle - call - put) < 2.0e-6f,
        "Bates straddle does not equal call plus put"
    );

    float gap_call = 0.0f;
    float gap_put = 0.0f;
    price_one(
        model,
        product::GapOptionParameters{1.0f, 1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_gap_option_cuda<OptionSide::call>,
        gap_call,
        error
    );
    price_one(
        model,
        product::GapOptionParameters{1.0f, 1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_gap_option_cuda<OptionSide::put>,
        gap_put,
        error
    );
    require(
        std::fabs(gap_call - call) < 2.0e-6f
            && std::fabs(gap_put - put) < 2.0e-6f,
        "Zero-gap Bates options do not match vanilla prices"
    );

    float digital_call = 0.0f;
    float digital_put = 0.0f;
    price_one(
        model,
        product::DigitalOptionParameters{1.0f, 252U, 1.0f},
        ai_factory::workbench::model::equity::bates::launch_bates_digital_option_cuda<OptionSide::call>,
        digital_call,
        error
    );
    price_one(
        model,
        product::DigitalOptionParameters{1.0f, 252U, 1.0f},
        ai_factory::workbench::model::equity::bates::launch_bates_digital_option_cuda<OptionSide::put>,
        digital_put,
        error
    );
    require(
        std::fabs(digital_call + digital_put - std::exp(-0.02f)) < 2.0e-6f,
        "Digital call and put do not sum to discounted cash"
    );

    float asset_call = 0.0f;
    float asset_put = 0.0f;
    price_one(
        model,
        product::AssetOrNothingOptionParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_asset_or_nothing_option_cuda<OptionSide::call>,
        asset_call,
        error
    );
    price_one(
        model,
        product::AssetOrNothingOptionParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_asset_or_nothing_option_cuda<OptionSide::put>,
        asset_put,
        error
    );

    float asian_put = 0.0f;
    price_one(
        model,
        product::AsianOptionParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_asian_option_cuda<OptionSide::put>,
        asian_put,
        error
    );

    float geometric_call = 0.0f;
    float geometric_put = 0.0f;
    price_one(
        model,
        product::GeometricAsianOptionParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_geometric_asian_option_cuda<OptionSide::call>,
        geometric_call,
        error
    );
    price_one(
        model,
        product::GeometricAsianOptionParameters{1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_geometric_asian_option_cuda<OptionSide::put>,
        geometric_put,
        error
    );

    float forward_call = 0.0f;
    float forward_put = 0.0f;
    price_one(
        model,
        product::ForwardStartOptionParameters{1.0f, 126U, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_forward_start_option_cuda<OptionSide::call>,
        forward_call,
        error
    );
    price_one(
        model,
        product::ForwardStartOptionParameters{1.0f, 126U, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_forward_start_option_cuda<OptionSide::put>,
        forward_put,
        error
    );

    float up_and_out_call = 0.0f;
    float up_and_in_call = 0.0f;
    float down_and_out_put = 0.0f;
    float down_and_in_put = 0.0f;
    float double_knock_out_call = 0.0f;
    float double_knock_out_put = 0.0f;
    float up_and_out_call_error = 0.0f;
    float up_and_in_call_error = 0.0f;
    float down_and_out_put_error = 0.0f;
    float down_and_in_put_error = 0.0f;
    float double_knock_out_call_error = 0.0f;
    float double_knock_out_put_error = 0.0f;
    price_one(
        model,
        product::UpAndOutOptionParameters{1.0f, 1.2f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_up_and_out_option_cuda<OptionSide::call>,
        up_and_out_call,
        up_and_out_call_error
    );
    price_one(
        model,
        product::DownAndOutOptionParameters{1.0f, 0.8f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_down_and_out_option_cuda<OptionSide::put>,
        down_and_out_put,
        down_and_out_put_error
    );
    price_one(
        model,
        product::UpAndInOptionParameters{1.0f, 1.2f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_up_and_in_option_cuda<OptionSide::call>,
        up_and_in_call,
        up_and_in_call_error
    );
    price_one(
        model,
        product::DownAndInOptionParameters{1.0f, 0.8f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_down_and_in_option_cuda<OptionSide::put>,
        down_and_in_put,
        down_and_in_put_error
    );
    price_one(
        model,
        product::DoubleKnockOutOptionParameters{1.0f, 0.8f, 1.2f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_double_knock_out_option_cuda<OptionSide::call>,
        double_knock_out_call,
        double_knock_out_call_error
    );
    price_one(
        model,
        product::DoubleKnockOutOptionParameters{1.0f, 0.8f, 1.2f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_double_knock_out_option_cuda<OptionSide::put>,
        double_knock_out_put,
        double_knock_out_put_error
    );
    require(
        std::isfinite(asset_call) && std::isfinite(asset_put)
            && std::isfinite(asian_put)
            && std::isfinite(geometric_call)
            && std::isfinite(geometric_put)
            && std::isfinite(forward_call)
            && std::isfinite(forward_put)
            && error > 0.0f && call_error > 0.0f && put_error > 0.0f
            && up_and_out_call_error > 0.0f
            && up_and_in_call_error > 0.0f
            && down_and_out_put_error > 0.0f
            && down_and_in_put_error > 0.0f
            && double_knock_out_call_error > 0.0f
            && double_knock_out_put_error > 0.0f,
        "Bates terminal-payoff launcher returned invalid statistics"
    );
    require(
        up_and_out_call <= call
                + six_sigma(up_and_out_call_error, call_error)
            && double_knock_out_call <= call
                + six_sigma(double_knock_out_call_error, call_error)
            && down_and_out_put <= put
                + six_sigma(down_and_out_put_error, put_error)
            && double_knock_out_put <= put
                + six_sigma(double_knock_out_put_error, put_error),
        "Bates knock-out price exceeds its vanilla price"
    );
    require(
        std::fabs(up_and_in_call + up_and_out_call - call)
            < six_sigma(
                up_and_in_call_error, up_and_out_call_error, call_error
            ),
        "Bates up-in plus up-out does not equal the vanilla call"
    );
    require(
        std::fabs(down_and_in_put + down_and_out_put - put)
            < six_sigma(
                down_and_in_put_error, down_and_out_put_error, put_error
            ),
        "Bates down-in plus down-out does not equal the vanilla put"
    );

    float up_one_touch = 0.0f;
    float up_no_touch = 0.0f;
    price_one(
        model,
        product::UpOneTouchParameters{1.2f, 1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_up_one_touch_cuda,
        up_one_touch,
        error
    );
    price_one(
        model,
        product::UpNoTouchParameters{1.2f, 1.0f, 252U},
        ai_factory::workbench::model::equity::bates::launch_bates_up_no_touch_cuda,
        up_no_touch,
        error
    );
    require(
        std::fabs(up_one_touch + up_no_touch - std::exp(-0.02f)) < 3.0e-6f,
        "Bates one-touch plus no-touch does not equal discounted cash"
    );
}
