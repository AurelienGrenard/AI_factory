// Compare mixed-speed G2/G2++ moments with independent host kernel integration.
#include "common/check_cuda.cuh"
#include "model/fixed_income/g2/analytics_impl.cuh"
#include "model/fixed_income/g2_plus_plus/nelson_siegel/analytics_impl.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
namespace wb = ai_factory::workbench;
namespace g2 = wb::model::fixed_income::g2;
namespace fitted = wb::model::fixed_income::g2_plus_plus::nelson_siegel;
struct Input { g2::ProcessParameters process; float delta; };
struct Result { float covariance[6]; float variance; float log_bond; float fitted_log_bond; };

__global__ void evaluate(const Input* inputs, Result* outputs, unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto input = inputs[index];
    const auto transition = g2::joint::prepare_transition(g2::prepare_model(input.process), input.delta);
    const float x = transition.state_x_standard_deviation;
    const float y0 = transition.state_y_x_normal_loading;
    const float y1 = transition.state_y_independent_standard_deviation;
    const float i0 = transition.integral_x_normal_loading;
    const float i1 = transition.integral_y_normal_loading;
    const float i2 = transition.integral_independent_standard_deviation;
    const g2::State state{0.02f, 0.01f};
    const auto model = fitted::compose_fitted_model({input.process}, {0.03f, 0.0f, 0.0f, 1.0f});
    outputs[index] = {
        {x*x, x*y0, y0*y0+y1*y1, x*i0, y0*i0+y1*i1, i0*i0+i1*i1+i2*i2},
        g2::integral_moments(input.process, input.delta).variance,
        g2::log_zero_coupon_bond({input.process, state}, state, 0.0f, input.delta),
        fitted::log_zero_coupon_bond(model, state, input.delta, 2.0f*input.delta),
    };
}

long double loading(long double rate, long double time) {
    return -std::expm1(-rate*time) / rate;
}

std::array<long double, 6> covariance(const Input& input, long double duration) {
    // Simpson integration of deterministic Brownian kernels, not a copied
    // closed-form expression. FP64/long double are host-only in this test.
    std::array<long double, 6> result{};
    constexpr unsigned int intervals = 2048U;
    const auto& p = input.process;
    const long double sx = p.volatility_x, sy = p.volatility_y, rho = p.correlation;
    for (unsigned int step = 0; step <= intervals; ++step) {
        const long double t = duration * step / intervals;
        const long double x = sx * std::exp(-p.mean_reversion_x*t);
        const long double y = sy * std::exp(-p.mean_reversion_y*t);
        const long double ix = sx * loading(p.mean_reversion_x, t);
        const long double iy = sy * loading(p.mean_reversion_y, t);
        const std::array<long double, 6> values{
            x*x, rho*x*y, y*y, x*(ix+rho*iy), y*(iy+rho*ix),
            ix*ix+iy*iy+2*rho*ix*iy,
        };
        const int weight = step == 0 || step == intervals ? 1 : step % 2 ? 4 : 2;
        for (unsigned int component = 0; component < 6; ++component) {
            result[component] += weight * values[component];
        }
    }
    for (auto& value : result) value *= duration / (3*intervals);
    return result;
}
}  // namespace

int main() {
    using wb::check_cuda;
    int count = 0;
    const auto availability = cudaGetDeviceCount(&count);
    if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver || count == 0) return 77;
    check_cuda(availability, "G2 covariance device discovery");
    std::vector<Input> inputs;
    for (const float slow : {1.0e-8f, 1.0e-6f, 1.0e-4f, 0.001f, 0.01f, 0.019f, 0.021f, 0.1f, 0.124f, 0.125f, 0.126f}) {
        for (const float fast : {0.1f, 0.124f, 0.125f, 0.126f, 0.499f, 0.5f, 0.501f, 0.999f, 1.0f, 1.001f, 10.0f}) {
            for (const float duration : {0.01f, 0.1f, 1.0f, 5.0f}) {
                for (const float rho : {-0.9f, 0.0f, 0.9f}) {
                    inputs.push_back({{slow, 0.2f, fast, 0.2f, rho}, duration});
                    inputs.push_back({{fast, 0.2f, slow, 0.2f, rho}, duration});
                }
            }
        }
    }
    Input* device_inputs = nullptr;
    Result* device_outputs = nullptr;
    std::vector<Result> results(inputs.size());
    check_cuda(cudaMalloc(&device_inputs, inputs.size()*sizeof(Input)), "G2 inputs allocation");
    check_cuda(cudaMalloc(&device_outputs, results.size()*sizeof(Result)), "G2 outputs allocation");
    check_cuda(cudaMemcpy(device_inputs, inputs.data(), inputs.size()*sizeof(Input), cudaMemcpyHostToDevice), "G2 input copy");
    evaluate<<<(inputs.size()+127U)/128U, 128U>>>(device_inputs, device_outputs, inputs.size());
    check_cuda(cudaGetLastError(), "G2 covariance launch");
    check_cuda(cudaMemcpy(results.data(), device_outputs, results.size()*sizeof(Result), cudaMemcpyDeviceToHost), "G2 result copy");
    check_cuda(cudaFree(device_outputs), "G2 outputs release");
    check_cuda(cudaFree(device_inputs), "G2 inputs release");
    long double worst = 0;
    for (std::size_t row = 0; row < inputs.size(); ++row) {
        const auto& input = inputs[row];
        const auto expected = covariance(input, input.delta);
        const auto doubled = covariance(input, 2.0L*input.delta);
        const std::array<long double, 6> scales{expected[0], std::sqrt(expected[0]*expected[2]),
            expected[2], std::sqrt(expected[0]*expected[5]), std::sqrt(expected[2]*expected[5]), expected[5]};
        for (unsigned int component = 0; component < 6; ++component) {
            const auto error = std::abs(results[row].covariance[component]-expected[component]) / scales[component];
            worst = std::max(worst, error);
            if (!std::isfinite(error) || error > 3.0e-5L) {
                std::cerr << "row=" << row << " a=" << input.process.mean_reversion_x
                          << " b=" << input.process.mean_reversion_y << " dt=" << input.delta
                          << " rho=" << input.process.correlation << " component=" << component
                          << " normalized_error=" << error << '\n';
                throw std::runtime_error("G2 reconstructed covariance disagrees with kernel integration");
            }
        }
        const long double state_term = -0.02f*loading(input.process.mean_reversion_x, input.delta)
            - 0.01f*loading(input.process.mean_reversion_y, input.delta);
        const long double log_bond = state_term + 0.5L*expected[5];
        const long double fitted_log_bond = state_term - 0.03f*input.delta
            + expected[5] - 0.5L*doubled[5];
        if (!(std::abs(results[row].variance-expected[5]) <= 3.0e-5L*expected[5])
            || !(std::abs(results[row].log_bond-log_bond) <= 3.0e-5L*(1+std::abs(log_bond)))
            || !(std::abs(results[row].fitted_log_bond-fitted_log_bond) <= 3.0e-5L*(1+std::abs(fitted_log_bond)))) {
            throw std::runtime_error("G2/G2++ bond moments disagree with kernel integration");
        }
    }
    std::cout << inputs.size() << " mixed-speed rows; worst normalized covariance error=" << worst << '\n';
}
