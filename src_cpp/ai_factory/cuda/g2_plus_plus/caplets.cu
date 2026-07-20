#include "ai_factory/cuda/g2_plus_plus/caplets.cuh"

#include "ai_factory/common/fixed_income/g2_plus_plus.hpp"
#include "ai_factory/cuda/common/philox.cuh"
#include "ai_factory/cuda/common/reductions.cuh"
#include "ai_factory/cuda/common/runtime.cuh"

namespace ai_factory::cuda {
namespace {

struct Tag {};

struct Coefficients {
    fixed_income::G2Transition transition;
    double bond_a;
    double bond_x;
    double bond_y;
    double discount_a;
};

__global__ void kernel(
    const G2PlusPlusCapletRow* rows,
    std::size_t count,
    std::size_t paths,
    MonteCarloOutput* outputs
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x);
    if (row_index >= count) {
        return;
    }
    const auto row = rows[row_index];
    const auto& model = row.model;
    const auto& product = row.product;
    __shared__ Coefficients coefficients;
    if (threadIdx.x == 0) {
        coefficients.transition = fixed_income::make_g2_transition(
            model.mean_reversion_x,
            model.volatility_x,
            model.mean_reversion_y,
            model.volatility_y,
            model.rho,
            product.fixing_time
        );
        const double maturity = product.fixing_time + product.accrual_period;
        coefficients.bond_a = fixed_income::g2_bond_price(
            0.0, 0.0, product.fixing_time, maturity,
            model.mean_reversion_x, model.volatility_x,
            model.mean_reversion_y, model.volatility_y, model.rho,
            row.curve.beta0, row.curve.beta1, row.curve.beta2, row.curve.tau
        );
        coefficients.bond_x = fixed_income::ou_b(
            model.mean_reversion_x, product.accrual_period
        );
        coefficients.bond_y = fixed_income::ou_b(
            model.mean_reversion_y, product.accrual_period
        );
        coefficients.discount_a = fixed_income::nelson_siegel_discount(
            product.fixing_time,
            row.curve.beta0, row.curve.beta1, row.curve.beta2, row.curve.tau
        ) * exp(-0.5 * fixed_income::g2_integral_variance(
            model.mean_reversion_x, model.volatility_x,
            model.mean_reversion_y, model.volatility_y, model.rho,
            product.fixing_time
        ));
    }
    __syncthreads();

    double sum = 0.0;
    double sumsq = 0.0;
    for (std::size_t path = threadIdx.x; path < paths; path += blockDim.x) {
        const auto normals = rng::standard_normal_quad(row.seed, 0U, path);
        double x = 0.0;
        double y = 0.0;
        double integrated_x = 0.0;
        double integrated_y = 0.0;
        fixed_income::apply_g2_transition(
            coefficients.transition,
            normals.first, normals.second, normals.third, normals.fourth,
            x, y, integrated_x, integrated_y
        );
        const double bond = coefficients.bond_a * exp(
            -coefficients.bond_x * x - coefficients.bond_y * y
        );
        const double discount = coefficients.discount_a
                                * exp(-integrated_x - integrated_y);
        const double payoff = discount * product.notional * fmax(
            1.0 - (1.0 + product.accrual_period * product.strike) * bond,
            0.0
        );
        sum += payoff;
        sumsq += payoff * payoff;
    }
    reductions::reduce_block(sum, sumsq);
    if (threadIdx.x == 0) {
        extern __shared__ double shared[];
        const double count_as_double = static_cast<double>(paths);
        const double mean = shared[0] / count_as_double;
        const double variance = (
            shared[blockDim.x] - count_as_double * mean * mean
        ) / (count_as_double - 1.0);
        outputs[row_index] = {
            mean,
            sqrt(fmax(variance, 0.0) / count_as_double),
        };
    }
}

}  // namespace

void price_g2_plus_plus_caplet_cuda(
    const G2PlusPlusCapletRow* host_rows,
    std::size_t count,
    std::size_t paths,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    auto& workspace = detail::reusable_cuda_workspace<Tag, 2>();
    auto* rows = workspace.buffer<G2PlusPlusCapletRow>(0, count, "caplet rows");
    auto* outputs = workspace.buffer<MonteCarloOutput>(1, count, "caplet outputs");
    detail::check_cuda(
        cudaMemcpy(rows, host_rows, count * sizeof(*host_rows), cudaMemcpyHostToDevice),
        "copy rows"
    );
    detail::check_cuda(cudaEventRecord(workspace.start_event()), "start");
    kernel<<<
        static_cast<unsigned>(count),
        detail::kThreadsPerBlock,
        2 * detail::kThreadsPerBlock * sizeof(double)
    >>>(rows, count, paths, outputs);
    detail::check_cuda(cudaGetLastError(), "G2++ caplet kernel");
    detail::check_cuda(cudaEventRecord(workspace.stop_event()), "stop");
    detail::check_cuda(cudaEventSynchronize(workspace.stop_event()), "sync");
    float milliseconds = 0.0F;
    detail::check_cuda(
        cudaEventElapsedTime(
            &milliseconds, workspace.start_event(), workspace.stop_event()
        ),
        "elapsed"
    );
    detail::check_cuda(
        cudaMemcpy(
            host_outputs, outputs, count * sizeof(*host_outputs),
            cudaMemcpyDeviceToHost
        ),
        "copy outputs"
    );
    if (timing != nullptr) {
        timing->simulation_ms = milliseconds;
        timing->total_ms = milliseconds;
    }
}

}  // namespace ai_factory::cuda
