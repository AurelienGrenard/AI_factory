#include "ai_factory/cuda/common/terminal_pricing.cuh"
#include "ai_factory/cuda/heston/api.cuh"
#include "ai_factory/cuda/heston/dynamics.cuh"

namespace ai_factory::cuda {
namespace {
struct WorkspaceTag {};
struct Simulator {
    __device__ __forceinline__ double operator()(
        const HestonRow& row, std::size_t path, std::size_t steps
    ) const {
        return heston_detail::simulate_heston_terminal_spot(row, path, steps);
    }
};
}

void price_heston_european_call_cuda(
    const HestonRow* rows, std::size_t row_count, std::size_t num_paths,
    std::size_t num_steps, MonteCarloOutput* outputs, CudaTiming* timing
) {
    terminal_pricing::run<WorkspaceTag, HestonRow, Simulator, terminal_pricing::Payoff::Call>(
        rows, row_count, num_paths, num_steps, outputs, timing
    );
}
}  // namespace ai_factory::cuda
