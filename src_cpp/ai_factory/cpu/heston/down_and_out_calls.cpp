#include "ai_factory/cpu/heston/down_and_out_calls.hpp"
#include "ai_factory/cpu/heston/barrier_calls.hpp"

namespace ai_factory::cpu::heston {

void price_down_and_out_call_batch(const cuda::HestonBarrierRow* rows, std::size_t row_count,
                       std::size_t num_paths, std::size_t num_steps,
                       cuda::MonteCarloOutput* outputs) {
    price_barrier_call_batch<false, false>(
        rows, row_count, num_paths, num_steps, outputs
    );
}

}  // namespace ai_factory::cpu::heston
