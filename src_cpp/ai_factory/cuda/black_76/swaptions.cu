#include "ai_factory/cuda/black_76/swaptions.cuh"

#include "ai_factory/common/fixed_income/black_76.hpp"
#include "ai_factory/common/fixed_income/nelson_siegel.hpp"
#include "ai_factory/cuda/common/runtime.cuh"

namespace ai_factory::cuda {
namespace {

struct Black76SwaptionWorkspaceTag {};

__global__ void swaption_kernel(
    const Black76SwaptionRow* rows,
    std::size_t row_count,
    MonteCarloOutput* outputs
) {
    const auto index = static_cast<std::size_t>(
        blockIdx.x * blockDim.x + threadIdx.x
    );
    if (index >= row_count) return;
    const auto row = rows[index];
    const auto product = row.product;
    const double discount_start = fixed_income::nelson_siegel_discount(
        product.expiry, row.curve.beta0, row.curve.beta1,
        row.curve.beta2, row.curve.tau
    );
    double annuity = 0.0;
    double discount_end = discount_start;
    for (int payment = 1; payment <= product.payment_count; ++payment) {
        discount_end = fixed_income::nelson_siegel_discount(
            product.expiry + payment * product.accrual_period,
            row.curve.beta0, row.curve.beta1,
            row.curve.beta2, row.curve.tau
        );
        annuity += product.accrual_period * discount_end;
    }
    const double forward_swap = (
        discount_start - discount_end
    ) / annuity;
    const double option = fixed_income::shifted_black_option(
        forward_swap, product.fixed_rate, row.model.displacement,
        row.model.volatility * sqrt(product.expiry), product.direction
    );
    outputs[index] = {product.notional * annuity * option, 0.0};
}

}  // namespace

void price_black_76_swaption_cuda(
    const Black76SwaptionRow* host_rows,
    std::size_t row_count,
    MonteCarloOutput* host_outputs,
    CudaTiming* timing
) {
    auto& workspace = detail::reusable_cuda_workspace<
        Black76SwaptionWorkspaceTag, 2U
    >();
    auto* rows = workspace.buffer<Black76SwaptionRow>(0U, row_count, "swaption rows");
    auto* outputs = workspace.buffer<MonteCarloOutput>(1U, row_count, "swaption outputs");
    detail::check_cuda(cudaMemcpy(rows, host_rows, row_count * sizeof(*host_rows), cudaMemcpyHostToDevice), "copy swaption rows");
    detail::check_cuda(cudaEventRecord(workspace.start_event()), "swaption start");
    const unsigned threads = detail::kThreadsPerBlock;
    const unsigned blocks = static_cast<unsigned>((row_count + threads - 1U) / threads);
    swaption_kernel<<<blocks, threads>>>(rows, row_count, outputs);
    detail::check_cuda(cudaGetLastError(), "Black-76 swaption kernel");
    detail::check_cuda(cudaEventRecord(workspace.stop_event()), "swaption stop");
    detail::check_cuda(cudaEventSynchronize(workspace.stop_event()), "swaption sync");
    float milliseconds = 0.0F;
    detail::check_cuda(cudaEventElapsedTime(&milliseconds, workspace.start_event(), workspace.stop_event()), "swaption elapsed");
    detail::check_cuda(cudaMemcpy(host_outputs, outputs, row_count * sizeof(*host_outputs), cudaMemcpyDeviceToHost), "copy swaption outputs");
    if (timing) timing->simulation_ms = timing->total_ms = milliseconds;
}

}  // namespace ai_factory::cuda
