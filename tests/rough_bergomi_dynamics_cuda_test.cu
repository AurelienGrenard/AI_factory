// Check the shared fractional driver and the two Gaussian-Volterra path maps.
#include "common/check_cuda.cuh"
#include "common/volterra/fractional_hybrid_driver.cuh"
#include "model/equity/rough/rough_bergomi/dynamics_impl.cuh"
#include "model/equity/rough/rough_sabr/dynamics_impl.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace {

namespace bergomi =
    ai_factory::workbench::model::equity::rough_bergomi;
namespace sabr = ai_factory::workbench::model::equity::rough_sabr;

struct Results {
    float first_weight;
    float terminal_driver_variance;
    float bergomi_log_spot;
    float bergomi_volatility;
    float sabr_log_spot;
    float sabr_volatility;
    float cev_log_spot;
};

__global__ void evaluate_dynamics(Results* output) {
    using Driver =
        ai_factory::workbench::volterra::FractionalHybridDriverPolicy;
    if (threadIdx.x != 0U || blockIdx.x != 0U) return;
    constexpr float dt = 1.0f / 360.0f;
    const Driver::PreparedDriver driver = Driver::prepare(0.10f, dt);
    const bergomi::ModelParameters bergomi_parameters = {
        1.0f, 0.03f, 0.01f, 0.04f, 1.2f, 0.10f, -0.70f,
    };
    const sabr::ModelParameters lognormal_sabr_parameters = {
        1.0f, 0.03f, 0.01f, 0.04f, 1.2f, 0.10f, -0.70f, 1.0f,
    };
    const sabr::ModelParameters cev_sabr_parameters = {
        1.0f, 0.03f, 0.01f, 0.04f, 1.2f, 0.10f, -0.70f, 0.70f,
    };
    const bergomi::PreparedModel bergomi_model =
        bergomi::prepare_model(bergomi_parameters, dt);
    const sabr::PreparedModel sabr_model =
        sabr::prepare_model(lognormal_sabr_parameters, dt);
    const sabr::PreparedModel cev_model =
        sabr::prepare_model(cev_sabr_parameters, dt);
    bergomi::State bergomi_state = bergomi::initial_state(bergomi_model);
    sabr::State sabr_state = sabr::initial_state(sabr_model);
    sabr::State cev_state = sabr::initial_state(cev_model);

    float increments[16]{};
    for (unsigned int step = 0U; step < 16U; ++step) {
        const float rough_normal = 0.07f * static_cast<float>(step) - 0.4f;
        const float singular_normal = 0.3f - 0.02f * step;
        const float spot_normal = -0.2f + 0.04f * step;
        increments[step] = driver.sqrt_time_step * rough_normal;
        float far = 0.0f;
        for (unsigned int previous = 0U; previous < step; ++previous) {
            far = fmaf(
                Driver::far_cell_weight(driver, step + 1U - previous),
                increments[previous],
                far
            );
        }
        const float value = Driver::value(
            driver,
            far,
            rough_normal,
            singular_normal
        );
        const float variance = Driver::variance(
            driver,
            static_cast<float>(step + 1U) * dt
        );
        bergomi::advance(
            bergomi_model,
            value,
            variance,
            rough_normal,
            spot_normal,
            bergomi_state
        );
        sabr::advance(
            sabr_model,
            value,
            variance,
            rough_normal,
            spot_normal,
            sabr_state
        );
        sabr::advance(
            cev_model,
            value,
            variance,
            rough_normal,
            spot_normal,
            cev_state
        );
    }
    *output = {
        Driver::far_cell_weight(driver, 2U),
        Driver::variance(driver, 16.0f * dt),
        bergomi_state.log_spot,
        sqrtf(bergomi_state.variance),
        sabr_state.log_spot,
        sabr_state.volatility,
        cev_state.log_spot,
    };
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    using ai_factory::workbench::check_cuda;
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "rough Volterra dynamics cudaGetDeviceCount");

    Results* device_results = nullptr;
    check_cuda(cudaMalloc(&device_results, sizeof(Results)), "dynamics malloc");
    evaluate_dynamics<<<1U, 1U>>>(device_results);
    check_cuda(cudaGetLastError(), "rough Volterra dynamics kernel");
    Results results{};
    check_cuda(
        cudaMemcpy(
            &results,
            device_results,
            sizeof(Results),
            cudaMemcpyDeviceToHost
        ),
        "rough Volterra dynamics copy"
    );
    check_cuda(cudaFree(device_results), "rough Volterra dynamics free");

    require(results.first_weight > 0.0f, "hybrid far weight is invalid");
    require(
        std::fabs(results.terminal_driver_variance
                  - std::pow(16.0f / 360.0f, 0.2f)) < 2.0e-6f,
        "fractional driver variance has the wrong normalization"
    );
    require(
        std::fabs(results.bergomi_log_spot - results.sabr_log_spot) < 2.0e-6f
            && std::fabs(
                results.bergomi_volatility - results.sabr_volatility
            ) < 2.0e-6f,
        "rough SABR beta=1 does not reduce to rough Bergomi"
    );
    require(
        std::isfinite(results.cev_log_spot),
        "rough SABR CEV path produced a non-finite state"
    );
    return 0;
}
