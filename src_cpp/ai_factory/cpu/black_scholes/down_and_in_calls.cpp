#include "ai_factory/cpu/black_scholes/down_and_in_calls.hpp"
#include "ai_factory/cpu/black_scholes/barrier_calls.hpp"

namespace ai_factory::cpu::black_scholes {

void price_down_and_in_call_batch(const cuda::BlackScholesBarrierRow* rows, std::size_t row_count,
                       std::size_t num_paths, std::size_t num_steps,
                       cuda::MonteCarloOutput* outputs) {
    price_barrier_call_batch<false, true>(
        rows, row_count, num_paths, num_steps, outputs
    );
}

}  // namespace ai_factory::cpu::black_scholes
