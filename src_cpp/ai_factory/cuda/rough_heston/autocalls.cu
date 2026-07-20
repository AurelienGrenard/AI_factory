#include "ai_factory/cuda/rough_heston/autocalls.cuh"
#include "ai_factory/cuda/common/autocall_reduction.cuh"
#include "ai_factory/cuda/common/runtime.cuh"
#include "ai_factory/cuda/rough_heston/api.cuh"
#include "ai_factory/cuda/rough_heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ai_factory::cuda {
namespace {
constexpr int kThreads = 128;
struct WorkspaceTag {};

__global__ void partial_kernel(
    const RoughHestonAutocallRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    std::size_t blocks_per_row, double* partials
) {
    const auto row_index = static_cast<std::size_t>(blockIdx.x) / blocks_per_row;
    const auto block = static_cast<std::size_t>(blockIdx.x) % blocks_per_row;
    if (row_index >= row_count) return;
    const auto row = rows[row_index];
    const auto path = block * blockDim.x + threadIdx.x;
    autocall_detail::PathMetrics metrics{};
    if (path < num_paths) {
        autocall_detail::PathState state{};
        std::size_t call_observation = 0U;
        const auto stride = num_steps / row.product.observation_count;
        struct Observer {
            const RoughHestonAutocallRow& row;
            std::size_t stride;
            autocall_detail::PathState& state;
            std::size_t& call_observation;
            __device__ void operator()(std::size_t step, double spot, double) {
                if (state.called || (step + 1U) % stride != 0U) return;
                const auto observation = (step + 1U) / stride;
                const double time = row.model.maturity
                    * static_cast<double>(observation)
                    / static_cast<double>(row.product.observation_count);
                if (autocall_detail::observe(
                    row.product, spot / row.model.spot, observation, time,
                    row.model.risk_free_rate, state
                )) call_observation = observation;
            }
        } observer{row, stride, state, call_observation};
        const double terminal = rough_heston_detail::simulate(
            row.model, path, num_steps, observer
        );
        metrics = autocall_detail::finish(
            row.product, terminal / row.model.spot, row.model.maturity,
            row.model.risk_free_rate, call_observation, state
        );
    }
    double values[autocall_detail::kMetricCount]{};
    autocall_detail::metrics_to_values(metrics, values);
    autocall_detail::reduce_and_store(
        values, row_index * blocks_per_row + block,
        row_count * blocks_per_row, partials
    );
}
}

void price_rough_heston_autocall_cuda(
    const RoughHestonAutocallRow* host_rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps,
    AutocallOutput* host_outputs, CudaTiming* timing
) {
    if (!row_count || num_paths < 2U) throw std::invalid_argument("Invalid rough Heston autocall batch.");
    for (std::size_t i=0;i<row_count;++i) if (!host_rows[i].product.observation_count || num_steps % host_rows[i].product.observation_count) throw std::invalid_argument("Autocall observations must divide steps.");
    const auto blocks = (num_paths + kThreads - 1U) / kThreads;
    const auto partial_count = row_count * blocks;
    auto& workspace = detail::reusable_cuda_workspace<WorkspaceTag,3U>();
    auto* rows = workspace.buffer<RoughHestonAutocallRow>(0U,row_count,"rough Heston autocall rows");
    auto* outputs = workspace.buffer<AutocallOutput>(1U,row_count,"rough Heston autocall outputs");
    auto* partials = workspace.buffer<double>(2U,autocall_detail::kMetricCount*partial_count,"rough Heston autocall partials");
    detail::check_cuda(cudaMemcpy(rows,host_rows,row_count*sizeof(RoughHestonAutocallRow),cudaMemcpyHostToDevice),"copy rough Heston autocall rows");
    auto start=workspace.start_event(); auto stop=workspace.stop_event();
    detail::check_cuda(cudaEventRecord(start),"rough Heston autocall start");
    const auto shared=autocall_detail::kMetricCount*kThreads*sizeof(double);
    partial_kernel<<<static_cast<unsigned int>(partial_count),kThreads,shared>>>(rows,row_count,num_paths,num_steps,blocks,partials);
    detail::check_cuda(cudaGetLastError(),"rough Heston autocall partial");
    autocall_detail::autocall_finalize_kernel<<<static_cast<unsigned int>(row_count),kThreads,shared>>>(partials,row_count,blocks,num_paths,outputs);
    detail::check_cuda(cudaGetLastError(),"rough Heston autocall finalize");
    detail::check_cuda(cudaEventRecord(stop),"rough Heston autocall stop");
    detail::check_cuda(cudaEventSynchronize(stop),"rough Heston autocall sync");
    float ms=0.0F; detail::check_cuda(cudaEventElapsedTime(&ms,start,stop),"rough Heston autocall timing");
    detail::check_cuda(cudaMemcpy(host_outputs,outputs,row_count*sizeof(AutocallOutput),cudaMemcpyDeviceToHost),"copy rough Heston autocall outputs");
    if(timing)*timing={ms,ms};
}
}  // namespace ai_factory::cuda
