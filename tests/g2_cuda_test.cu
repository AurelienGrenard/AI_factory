// Validate exact G2 transitions and standalone affine analytics on CUDA.
#include "common/check_cuda.cuh"

// Include the implementation exactly as product kernels do.
#include "model/fixed_income/g2/analytics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr std::size_t kOutputCount = 24U;

// Evaluate transition covariance identities and bond-option parity.
__global__ void g2_test_kernel(float* outputs) {
    if (blockIdx.x != 0U || threadIdx.x != 0U) return;
    namespace g2 = ai_factory::workbench::model::fixed_income::g2;

    const ai_factory::workbench::model::fixed_income::g2::ModelParameters parameters = {
        {0.15f, 0.01f, 0.70f, 0.008f, -0.40f},
        {0.02f, 0.01f},
    };
    constexpr float delta = 0.5f;
    const ai_factory::workbench::model::fixed_income::g2::PreparedModel prepared_model =
        ai_factory::workbench::model::fixed_income::g2::prepare_model(parameters.process);
    const ai_factory::workbench::model::fixed_income::g2::PreparedTransition transition =
        ai_factory::workbench::model::fixed_income::g2::prepare_transition(prepared_model, delta);
    const ai_factory::workbench::model::fixed_income::g2::joint::PreparedTransition joint_transition =
        ai_factory::workbench::model::fixed_income::g2::joint::prepare_transition(prepared_model, delta);
    const ai_factory::workbench::model::fixed_income::g2::IntegralMoments moments = ai_factory::workbench::model::fixed_income::g2::integral_moments(
        parameters.process, delta
    );

    outputs[0] = transition.state_x_standard_deviation
        * transition.state_x_standard_deviation;
    outputs[1] = transition.state_x_standard_deviation
        * transition.state_y_x_normal_loading;
    outputs[2] = transition.state_y_x_normal_loading
        * transition.state_y_x_normal_loading
        + transition.state_y_independent_standard_deviation
            * transition.state_y_independent_standard_deviation;
    outputs[3] = joint_transition.state_x_standard_deviation
        * joint_transition.integral_x_normal_loading;
    outputs[4] = joint_transition.state_y_x_normal_loading
            * joint_transition.integral_x_normal_loading
        + joint_transition.state_y_independent_standard_deviation
            * joint_transition.integral_y_normal_loading;
    outputs[5] = joint_transition.integral_x_normal_loading
            * joint_transition.integral_x_normal_loading
        + joint_transition.integral_y_normal_loading
            * joint_transition.integral_y_normal_loading
        + joint_transition.integral_independent_standard_deviation
            * joint_transition.integral_independent_standard_deviation;
    outputs[6] = moments.state_x_loading;
    outputs[7] = moments.state_y_loading;
    outputs[8] = moments.variance;
    outputs[9] = ai_factory::workbench::model::fixed_income::g2::short_rate(
        parameters, parameters.initial_state, 0.0f
    );

    constexpr float expiry = 1.0f;
    constexpr float maturity = 2.0f;
    constexpr float strike = 0.95f;
    const float call = ai_factory::workbench::model::fixed_income::g2::zero_coupon_bond_call_price(
        parameters, parameters.initial_state, 0.0f, expiry, maturity, strike
    );
    const float put = ai_factory::workbench::model::fixed_income::g2::zero_coupon_bond_put_price(
        parameters, parameters.initial_state, 0.0f, expiry, maturity, strike
    );
    const float expiry_bond = ai_factory::workbench::model::fixed_income::g2::zero_coupon_bond(
        parameters, parameters.initial_state, 0.0f, expiry
    );
    const float underlying_bond = ai_factory::workbench::model::fixed_income::g2::zero_coupon_bond(
        parameters, parameters.initial_state, 0.0f, maturity
    );
    outputs[10] = call;
    outputs[11] = put;
    outputs[12] = call - put - (underlying_bond - strike * expiry_bond);

    const ai_factory::workbench::model::fixed_income::g2::ProcessParameters deterministic = {
        0.15f, 0.0f, 0.70f, 0.0f, 0.0f
    };
    const ai_factory::workbench::model::fixed_income::g2::PreparedModel deterministic_model =
        ai_factory::workbench::model::fixed_income::g2::prepare_model(deterministic);
    const ai_factory::workbench::model::fixed_income::g2::PreparedTransition deterministic_transition =
        ai_factory::workbench::model::fixed_income::g2::prepare_transition(deterministic_model, delta);
    outputs[13] = deterministic_transition.state_x_standard_deviation;
    outputs[14] = deterministic_transition.state_y_independent_standard_deviation;
    ai_factory::workbench::model::fixed_income::g2::joint::State terminal{parameters.initial_state, 0.0f};
    ai_factory::workbench::model::fixed_income::g2::joint::one_step_transition(
        joint_transition, 0.2f, -0.3f, 0.5f, terminal
    );
    outputs[15] = terminal.state_integral;

    constexpr float small_delta = 1.0e-4f;
    const ai_factory::workbench::model::fixed_income::g2::IntegralMoments small_moments = ai_factory::workbench::model::fixed_income::g2::integral_moments(
        parameters.process, small_delta
    );
    outputs[16] = small_moments.state_x_loading / small_delta;
    outputs[17] = small_moments.state_y_loading / small_delta;
    outputs[18] = small_moments.variance
        / (small_delta * small_delta * small_delta);
    outputs[19] = ai_factory::workbench::model::fixed_income::g2::A(parameters, 0.0f, maturity);
    const ai_factory::workbench::model::fixed_income::g2::TwoFactorAffineBondLoadings bond_loadings = ai_factory::workbench::model::fixed_income::g2::B(
        parameters, 0.0f, maturity
    );
    outputs[20] = bond_loadings.state_x;
    outputs[21] = bond_loadings.state_y;
    outputs[22] = ai_factory::workbench::model::fixed_income::g2::log_zero_coupon_bond(
        parameters, parameters.initial_state, 0.0f, maturity
    );
    outputs[23] = ai_factory::workbench::model::fixed_income::g2::log_A(parameters, 0.0f, maturity)
        - bond_loadings.state_x * parameters.initial_state.state_x
        - bond_loadings.state_y * parameters.initial_state.state_y;
}

// Stop immediately with a readable invariant name.
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// Return B(delta) in FP64.
double loading(double mean_reversion, double delta) {
    return -std::expm1(-mean_reversion * delta) / mean_reversion;
}

}  // namespace

// Confirm the exact covariance factorization and analytical identities.
int main() {
    using namespace ai_factory::workbench;
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) {
        return 77;
    }
    check_cuda(availability, "G2 test cudaGetDeviceCount");

    float* device_outputs = nullptr;
    check_cuda(
        cudaMalloc(&device_outputs, kOutputCount * sizeof(float)),
        "G2 test cudaMalloc"
    );
    float outputs[kOutputCount] = {};
    g2_test_kernel<<<1U, 1U>>>(device_outputs);
    check_cuda(cudaGetLastError(), "G2 test kernel");
    check_cuda(
        cudaMemcpy(
            outputs,
            device_outputs,
            sizeof(outputs),
            cudaMemcpyDeviceToHost
        ),
        "G2 test cudaMemcpy"
    );
    check_cuda(cudaFree(device_outputs), "G2 test cudaFree");

    constexpr double a = 0.15;
    constexpr double sigma = 0.01;
    constexpr double b = 0.70;
    constexpr double eta = 0.008;
    constexpr double rho = -0.40;
    constexpr double delta = 0.5;
    const double variance_x = sigma * sigma
        * (-std::expm1(-2.0 * a * delta)) / (2.0 * a);
    const double variance_y = eta * eta
        * (-std::expm1(-2.0 * b * delta)) / (2.0 * b);
    const double covariance_xy = rho * sigma * eta
        * (-std::expm1(-(a + b) * delta)) / (a + b);
    const double cross_integral = rho * sigma * eta / (a * b)
        * (delta - loading(a, delta) - loading(b, delta)
            + loading(a + b, delta));
    const auto integral_variance = [](double k, double vol, double d) {
        return vol * vol / (k * k)
            * (d - 2.0 * loading(k, d) + loading(2.0 * k, d));
    };
    const double total_integral_variance =
        integral_variance(a, sigma, delta)
        + integral_variance(b, eta, delta)
        + 2.0 * cross_integral;
    const double covariance_x_integral =
        0.5 * sigma * sigma * loading(a, delta) * loading(a, delta)
        + rho * sigma * eta
            * (loading(a, delta) - loading(a + b, delta)) / b;
    const double covariance_y_integral =
        0.5 * eta * eta * loading(b, delta) * loading(b, delta)
        + rho * sigma * eta
            * (loading(b, delta) - loading(a + b, delta)) / a;

    require(std::fabs(outputs[0] - variance_x) < 1.0e-8, "G2 Var[X] mismatch");
    require(std::fabs(outputs[1] - covariance_xy) < 1.0e-8, "G2 Cov[X,Y] mismatch");
    require(std::fabs(outputs[2] - variance_y) < 1.0e-8, "G2 Var[Y] mismatch");
    require(std::fabs(outputs[3] - covariance_x_integral) < 1.0e-8, "G2 Cov[X,I] mismatch");
    require(std::fabs(outputs[4] - covariance_y_integral) < 1.0e-8, "G2 Cov[Y,I] mismatch");
    require(std::fabs(outputs[5] - total_integral_variance) < 1.0e-8, "G2 integral variance mismatch");
    require(std::fabs(outputs[6] - loading(a, delta)) < 1.0e-6, "G2 X loading mismatch");
    require(std::fabs(outputs[7] - loading(b, delta)) < 1.0e-6, "G2 Y loading mismatch");
    require(std::fabs(outputs[8] - total_integral_variance) < 1.0e-8, "G2 moments mismatch");
    require(std::fabs(outputs[9] - 0.03f) < 1.0e-7f, "G2 short rate mismatch");
    require(outputs[10] >= 0.0f && outputs[11] >= 0.0f, "G2 option price invalid");
    require(std::fabs(outputs[12]) < 3.0e-6f, "G2 bond-option parity mismatch");
    require(outputs[13] == 0.0f && outputs[14] == 0.0f, "G2 deterministic transition is random");
    require(std::isfinite(outputs[15]), "G2 joint transition is non-finite");
    require(std::fabs(outputs[16] - 1.0f) < 1.0e-4f, "G2 small-time X loading mismatch");
    require(std::fabs(outputs[17] - 1.0f) < 1.0e-4f, "G2 small-time Y loading mismatch");
    const double instantaneous_variance =
        sigma * sigma + eta * eta + 2.0 * rho * sigma * eta;
    require(
        std::fabs(outputs[18] - instantaneous_variance / 3.0) < 1.0e-7,
        "G2 small-time integral variance mismatch"
    );
    require(
        outputs[19] > 0.0f && outputs[20] > 0.0f && outputs[21] > 0.0f
            && std::fabs(outputs[22] - outputs[23]) < 2.0e-7f,
        "G2 affine A/B decomposition is incorrect"
    );
}
