#include "ai_factory/cuda/rough_bergomi/api.cuh"
#include "ai_factory/cuda/rough_bergomi/terminal_pricing.cuh"

namespace ai_factory::cuda {
namespace { struct WorkspaceTag {}; }
void price_rough_bergomi_european_call_cuda(
    const RoughBergomiRow* rows, std::size_t row_count, std::size_t num_paths,
    std::size_t num_steps, MonteCarloOutput* outputs, CudaTiming* timing
) {
    rough_bergomi_terminal::run<WorkspaceTag, terminal_pricing::Payoff::Call>(
        rows, row_count, num_paths, num_steps, outputs, timing
    );
}
}  // namespace ai_factory::cuda
