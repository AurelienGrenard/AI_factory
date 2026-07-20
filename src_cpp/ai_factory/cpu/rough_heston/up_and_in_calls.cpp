#include "ai_factory/cpu/rough_heston/up_and_in_calls.hpp"
#include "ai_factory/cpu/rough_heston/barrier_calls.hpp"

namespace ai_factory::cpu::rough_heston {

void price_up_and_in_call_batch(const cuda::RoughHestonBarrierRow* rows, std::size_t row_count,
                       std::size_t num_paths, std::size_t num_steps,
                       cuda::MonteCarloOutput* outputs) {
    price_barrier_call_batch<true, true>(
        rows, row_count, num_paths, num_steps, outputs
    );
}

}  // namespace ai_factory::cpu::rough_heston
