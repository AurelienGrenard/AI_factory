#include "ai_factory/cuda/black_scholes/api.cuh"
#include "ai_factory/cuda/black_scholes/dynamics.cuh"
#include "ai_factory/cuda/common/terminal_pricing.cuh"

namespace ai_factory::cuda {
namespace {
struct WorkspaceTag {};
struct Simulator {
    __device__ __forceinline__ double operator()(
        const BlackScholesRow& row, std::size_t path, std::size_t steps
    ) const {
        return black_scholes_detail::simulate_terminal_spot(row, path, steps);
    }
};
}

void price_black_scholes_digital_call_cuda(
    const BlackScholesRow* rows, std::size_t row_count, std::size_t num_paths,
    std::size_t num_steps, MonteCarloOutput* outputs, CudaTiming* timing
) {
    terminal_pricing::run<WorkspaceTag, BlackScholesRow, Simulator, terminal_pricing::Payoff::DigitalCall>(
        rows, row_count, num_paths, num_steps, outputs, timing
    );
}
}  // namespace ai_factory::cuda
