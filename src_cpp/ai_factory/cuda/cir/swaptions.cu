#include "ai_factory/cuda/cir/swaptions.cuh"

#include "ai_factory/common/fixed_income/cir.hpp"
#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

#include <cmath>

namespace ai_factory::cuda {
namespace {

constexpr int kMaxPayments = 20;
constexpr double kQePsiCutoff = 1.5;
struct CirSwaptionWorkspaceTag {};
struct StepCoefficients {
    std::size_t num_steps;
    double dt;
    double decay;
    double one_minus_decay;
    double volatility_squared;
};

__global__ void swaption_kernel(
    const CirSwaptionRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    double target_dt,
    MonteCarloOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= row_count) {
        return;
    }
    const auto row = rows[row_index];
    const auto& model = row.model;
    const auto& product = row.product;

    __shared__ double bond_a[kMaxPayments];
    __shared__ double bond_b[kMaxPayments];
    __shared__ StepCoefficients coefficients;
    if (threadIdx.x < product.payment_count) {
        const double horizon =
            (static_cast<int>(threadIdx.x) + 1) * product.accrual_period;
        bond_a[threadIdx.x] = fixed_income::cir_bond_a(
            model.kappa, model.theta, model.volatility, horizon
        );
        bond_b[threadIdx.x] = fixed_income::cir_bond_b(
            model.kappa, model.volatility, horizon
        );
    }
    if (threadIdx.x == 0) {
        const auto rounded_steps = llround(product.expiry / target_dt);
        coefficients.num_steps = static_cast<std::size_t>(
            rounded_steps > 0LL ? rounded_steps : 1LL
        );
        coefficients.dt = product.expiry
            / static_cast<double>(coefficients.num_steps);
        coefficients.decay = exp(-model.kappa * coefficients.dt);
        coefficients.one_minus_decay = 1.0 - coefficients.decay;
        coefficients.volatility_squared = model.volatility * model.volatility;
    }
    __syncthreads();

    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (std::size_t path = static_cast<std::size_t>(threadIdx.x);
         path < num_paths;
         path += static_cast<std::size_t>(blockDim.x)) {
        double rate = model.initial_rate;
        double integral = 0.0;
        rng::NormalSequence normals(row.seed, path, 0U);
        rng::UniformSequence uniforms(row.seed, path + num_paths, 0U);

        for (std::size_t step = 0; step < coefficients.num_steps; ++step) {
            const double previous_rate = rate;
            const double mean =
                model.theta + (rate - model.theta) * coefficients.decay;
            const double variance =
                rate * coefficients.volatility_squared * coefficients.decay
                    * coefficients.one_minus_decay
                    / model.kappa
                + model.theta * coefficients.volatility_squared
                      * coefficients.one_minus_decay
                      * coefficients.one_minus_decay
                      / (2.0 * model.kappa);
            const double psi = variance / (mean * mean);
            const double normal = normals.next();
            const double uniform = uniforms.next();

            if (psi <= kQePsiCutoff) {
                const double inverse_psi = 1.0 / psi;
                const double b_squared =
                    2.0 * inverse_psi - 1.0
                    + sqrt(2.0 * inverse_psi)
                          * sqrt(fmax(2.0 * inverse_psi - 1.0, 0.0));
                const double scale = mean / (1.0 + b_squared);
                const double shifted = sqrt(b_squared) + normal;
                rate = scale * shifted * shifted;
            } else {
                const double probability = (psi - 1.0) / (psi + 1.0);
                const double beta = (1.0 - probability) / mean;
                rate = uniform <= probability
                           ? 0.0
                           : log(
                                 (1.0 - probability) / (1.0 - uniform)
                             )
                                 / beta;
            }
            integral += 0.5 * (previous_rate + rate) * coefficients.dt;
        }

        double annuity = 0.0;
        for (int payment = 0;
             payment < product.payment_count;
             ++payment) {
            annuity += product.accrual_period * bond_a[payment]
                       * exp(-bond_b[payment] * rate);
        }
        const int last_payment = product.payment_count - 1;
        const double end_bond = bond_a[last_payment]
                                * exp(-bond_b[last_payment] * rate);
        const double swap = static_cast<double>(product.direction)
                            * (1.0 - end_bond
                               - product.fixed_rate * annuity);
        const double payoff = exp(-integral) * product.notional
                              * fmax(swap, 0.0);
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

void price_cir_swaption_cuda(
    const CirSwaptionRow* host_rows,
    std::size_t row_count,
    std::size_t num_paths,
    double target_dt,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    auto& workspace = detail::reusable_cuda_workspace<
        CirSwaptionWorkspaceTag, 2U
    >();
    auto* device_rows = workspace.buffer<CirSwaptionRow>(
        0U, row_count, "cudaMalloc rows"
    );
    auto* device_outputs = workspace.buffer<MonteCarloOutput>(
        1U, row_count, "cudaMalloc outputs"
    );
    detail::check_cuda(
        cudaMemcpy(
            device_rows,
            host_rows,
            row_count * sizeof(CirSwaptionRow),
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
    >>>(
        device_rows,
        row_count,
        num_paths,
        target_dt,
        device_outputs
    );
    detail::check_cuda(cudaGetLastError(), "CIR swaption kernel");
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
