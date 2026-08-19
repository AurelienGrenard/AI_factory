// Validate G2++ curve fitting and affine analytics on CUDA.
#include "common/check_cuda.cuh"

// Include the implementation exactly as product kernels do.
#include "model/fixed_income/g2_plus_plus/nelson_siegel/analytics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr std::size_t kOutputCount = 13U;

// Evaluate initial-curve fitting and zero-coupon option parity.
__global__ void g2_plus_plus_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    namespace curve = ai_factory::workbench::curve::nelson_siegel;
    namespace g2pp = ai_factory::workbench::model::g2_plus_plus;
    namespace fitted = g2pp::nelson_siegel;

    const g2pp::G2PlusPlusModelParameters model = {{
        0.15f, 0.01f, 0.70f, 0.008f, -0.40f
    }};
    const curve::NelsonSiegelParameters initial_curve = {
        0.03f, -0.01f, 0.02f, 2.0f
    };
    const fitted::G2PlusPlusFittedParameters parameters =
        fitted::compose_model(model, initial_curve);
    const ai_factory::workbench::model::g2::G2State state{0.0f, 0.0f};
    constexpr float expiry = 1.0f;
    constexpr float maturity = 2.0f;
    constexpr float strike = 0.95f;

    outputs[0] = fitted::zero_coupon_bond(
        parameters, state, 0.0f, expiry
    );
    outputs[1] = curve::discount_factor(initial_curve, expiry);
    outputs[2] = fitted::zero_coupon_bond(
        parameters, state, 0.0f, maturity
    );
    outputs[3] = curve::discount_factor(initial_curve, maturity);
    outputs[4] = fitted::zero_coupon_bond_call_price(
        parameters, state, 0.0f, expiry, maturity, strike
    );
    outputs[5] = fitted::zero_coupon_bond_put_price(
        parameters, state, 0.0f, expiry, maturity, strike
    );
    outputs[6] = outputs[4] - outputs[5]
        - (outputs[2] - strike * outputs[0]);
    outputs[7] = fitted::short_rate(parameters, state, 0.0f);
    constexpr ai_factory::workbench::model::g2::G2State affine_state = {
        0.01f, -0.005f
    };
    outputs[8] = fitted::A(parameters, expiry, maturity);
    const ai_factory::workbench::model::g2::G2BondLoadings bond_loadings =
        fitted::B(parameters, expiry, maturity);
    outputs[9] = bond_loadings.state_x;
    outputs[10] = bond_loadings.state_y;
    outputs[11] = fitted::log_zero_coupon_bond(
        parameters, affine_state, expiry, maturity
    );
    outputs[12] = fitted::log_A(parameters, expiry, maturity)
        - bond_loadings.state_x * affine_state.state_x
        - bond_loadings.state_y * affine_state.state_y;
}

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

// Confirm exact curve fitting and the shared bond-option formula.
int main() {
    using namespace ai_factory::workbench;
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "G2++ test cudaGetDeviceCount");

    float* device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_outputs, kOutputCount * sizeof(float)),
        "G2++ test cudaMalloc"
    );
    float outputs[kOutputCount] = {};
    g2_plus_plus_test_kernel<<<1U, 1U>>>(device_outputs);
    check_cuda(cudaGetLastError(), "G2++ test kernel");
    check_cuda(
        cudaMemcpy(
            outputs,
            device_outputs,
            sizeof(outputs),
            cudaMemcpyDeviceToHost
        ),
        "G2++ test cudaMemcpy"
    );
    check_cuda(cudaFree(device_outputs), "G2++ test cudaFree");

    require(std::fabs(outputs[0] - outputs[1]) < 2.0e-6f, "G2++ expiry curve fit mismatch");
    require(std::fabs(outputs[2] - outputs[3]) < 2.0e-6f, "G2++ maturity curve fit mismatch");
    require(outputs[4] >= 0.0f && outputs[5] >= 0.0f, "G2++ option price invalid");
    require(std::fabs(outputs[6]) < 3.0e-6f, "G2++ bond-option parity mismatch");
    require(
        std::fabs(outputs[7] - 0.02f) < 2.0e-6f,
        "G2++ initial short rate does not match the initial forward"
    );
    require(
        outputs[8] > 0.0f && outputs[9] > 0.0f && outputs[10] > 0.0f
            && std::fabs(outputs[11] - outputs[12]) < 2.0e-7f,
        "G2++ affine A/B decomposition is incorrect"
    );
}
