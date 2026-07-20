#include "ai_factory/cuda/black_76/caplets.cuh"

#include "ai_factory/common/fixed_income/black_76.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"
#include "ai_factory/cuda/common/runtime.cuh"

namespace ai_factory::cuda {
namespace {

struct Black76CapletWorkspaceTag {};

__global__ void caplet_kernel(
    const Black76CapletRow* rows,
    std::size_t row_count,
    MonteCarloOutput* outputs
) {
    const auto index = static_cast<std::size_t>(
        blockIdx.x * blockDim.x + threadIdx.x
    );
    if (index >= row_count) return;
    const auto row = rows[index];
    const double fixing = row.product.fixing_time;
    const double payment = fixing + row.product.accrual_period;
    const double discount_fixing = fixed_income::nelson_siegel_discount(
        fixing, row.curve.beta0, row.curve.beta1,
        row.curve.beta2, row.curve.tau
    );
    const double discount_payment = fixed_income::nelson_siegel_discount(
        payment, row.curve.beta0, row.curve.beta1,
        row.curve.beta2, row.curve.tau
    );
    const double forward = (
        discount_fixing / discount_payment - 1.0
    ) / row.product.accrual_period;
    const double option = fixed_income::shifted_black_option(
        forward, row.product.strike, row.model.displacement,
        row.model.volatility * sqrt(fixing), 1
    );
    outputs[index] = {
        row.product.notional * row.product.accrual_period
            * discount_payment * option,
        0.0,
    };
}

}  // namespace

void price_black_76_caplet_cuda(
    const Black76CapletRow* host_rows,
    std::size_t row_count,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    auto& workspace = detail::reusable_cuda_workspace<
        Black76CapletWorkspaceTag, 2U
    >();
    auto* rows = workspace.buffer<Black76CapletRow>(0U, row_count, "caplet rows");
    auto* outputs = workspace.buffer<MonteCarloOutput>(1U, row_count, "caplet outputs");
    detail::check_cuda(cudaMemcpy(rows, host_rows, row_count * sizeof(*host_rows), cudaMemcpyHostToDevice), "copy caplet rows");
    detail::check_cuda(cudaEventRecord(workspace.start_event()), "caplet start");
    const unsigned threads = detail::kThreadsPerBlock;
    const unsigned blocks = static_cast<unsigned>((row_count + threads - 1U) / threads);
    caplet_kernel<<<blocks, threads>>>(rows, row_count, outputs);
    detail::check_cuda(cudaGetLastError(), "Black-76 caplet kernel");
    detail::check_cuda(cudaEventRecord(workspace.stop_event()), "caplet stop");
    detail::check_cuda(cudaEventSynchronize(workspace.stop_event()), "caplet sync");
    float milliseconds = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&milliseconds, workspace.start_event(), workspace.stop_event()), "caplet elapsed");
    detail::check_cuda(cudaMemcpy(host_outputs, outputs, row_count * sizeof(*host_outputs), cudaMemcpyDeviceToHost), "copy caplet outputs");
    if (timing) timing->simulation_ms = timing->total_ms = milliseconds;
}

}  // namespace ai_factory::cuda
