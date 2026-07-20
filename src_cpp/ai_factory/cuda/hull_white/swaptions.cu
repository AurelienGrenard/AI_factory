#include "ai_factory/cuda/hull_white/swaptions.cuh"

#include "ai_factory/common/fixed_income/hull_white.hpp"
#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cmath>

namespace ai_factory::cuda {
namespace {

constexpr int kMaxPayments = 20;
struct HullWhiteSwaptionWorkspaceTag {};
struct SwaptionCoefficients {
    double state_scale;
    double integral_loading;
    double residual_scale;
    double deterministic_integral;
};

__global__ void swaption_kernel(
    const HullWhiteSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    MonteCarloOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto row = rows[row_index];
    const auto& product = row.product;

    __shared__ double bond_a[kMaxPayments];
    __shared__ double bond_b[kMaxPayments];
    __shared__ SwaptionCoefficients coefficients;
    if (threadIdx.x < product.payment_count) {
        const int payment = static_cast<int>(threadIdx.x) + 1;
        const double maturity =
            product.expiry + payment * product.accrual_period;
        bond_a[threadIdx.x] = fixed_income::hull_white_bond_a(
            product.expiry,
            maturity,
            row.mean_reversion,
            row.volatility,
            row.beta0,
            row.beta1,
            row.beta2,
            row.tau
        );
        bond_b[threadIdx.x] = fixed_income::hull_white_b(
            row.mean_reversion, maturity - product.expiry
        );
    }
    if (threadIdx.x == 0) {
        const double state_variance = fixed_income::hull_white_state_variance(
            row.mean_reversion, row.volatility, product.expiry
        );
        const double integral_variance = fixed_income::hull_white_integral_variance(
            row.mean_reversion, row.volatility, product.expiry
        );
        const double covariance = fixed_income::hull_white_state_integral_covariance(
            row.mean_reversion, row.volatility, product.expiry
        );
        coefficients.state_scale = sqrt(state_variance);
        coefficients.integral_loading = covariance / coefficients.state_scale;
        coefficients.residual_scale = sqrt(fmax(
            integral_variance - coefficients.integral_loading
                * coefficients.integral_loading,
            0.0
        ));
        coefficients.deterministic_integral =
            fixed_income::hull_white_deterministic_integral(
                product.expiry, row.mean_reversion, row.volatility,
                row.beta0, row.beta1, row.beta2, row.tau
            );
    }
    __syncthreads();

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (std::size_t path = static_cast<std::size_t>(threadIdx.x);
         path < num_paths;
         path += static_cast<std::size_t>(blockDim.x)) {
        const auto normals = rng::standard_normal_pair(row.seed, 0U, path);
        const double state = coefficients.state_scale * normals.first;
        const double state_integral =
            coefficients.integral_loading * normals.first
            + coefficients.residual_scale * normals.second;

        double annuity = 0.0;
        for (int payment = 0;
             payment < product.payment_count;
             ++payment) {
            annuity += product.accrual_period * bond_a[payment]
                       * exp(-bond_b[payment] * state);
        }
        const int last_payment = product.payment_count - 1;
        const double end_bond = bond_a[last_payment]
                                * exp(-bond_b[last_payment] * state);
        const double swap = static_cast<double>(product.direction)
                            * (1.0 - end_bond
                               - product.fixed_rate * annuity);
        const double payoff = exp(-coefficients.deterministic_integral - state_integral)
                              * product.notional * fmax(swap, 0.0);
        local_sum += payoff;
        local_sumsq += payoff * payoff;
    }

    reductions::reduce_block(local_sum, local_sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const double path_count = static_cast<double>(num_paths);
        const double mean = shared[0] / path_count;
        const double variance =
            (shared[blockDim.x] - path_count * mean * mean)
            / (path_count - 1.0);
        outputs[row_index] = {
            mean,
            sqrt(fmax(variance, 0.0) / path_count),
        };
    }
}

}  // namespace

void price_hull_white_swaption_cuda(
    const HullWhiteSwaptionRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    auto& workspace = detail::reusable_cuda_workspace<
        HullWhiteSwaptionWorkspaceTag, 2U
    >();
    auto* device_rows = workspace.buffer<HullWhiteSwaptionRow>(
        0U, row_count, "cudaMalloc rows"
    );
    auto* device_outputs = workspace.buffer<MonteCarloOutput>(
        1U, row_count, "cudaMalloc outputs"
    );
    detail::check_cuda(
        cudaMemcpy(
            device_rows,
            host_rows,
            row_count * sizeof(HullWhiteSwaptionRow),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy rows"
    );

    const auto start = workspace.start_event();
    const auto stop = workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    swaption_kernel<<<
        static_cast<unsigned int>(row_count),
        detail::kThreadsPerBlock,
        2U * detail::kThreadsPerBlock * sizeof(double)
    >>>(device_rows, row_count, num_paths, device_outputs);
    detail::check_cuda(cudaGetLastError(), "Hull-White swaption kernel");
    detail::check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    detail::check_cuda(
        cudaEventSynchronize(stop), "cudaEventSynchronize stop"
    );
    float elapsed_ms = 0.0F;
    detail::check_cuda(
        cudaEventElapsedTime(&elapsed_ms, start, stop), "kernel timing"
    );
    detail::check_cuda(
        cudaMemcpy(
            host_outputs,
            device_outputs,
            row_count * sizeof(MonteCarloOutput),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy outputs"
    );
    if (timing != nullptr) {
        timing->simulation_ms = elapsed_ms;
        timing->total_ms = elapsed_ms;
    }
}

}  // namespace ai_factory::cuda
