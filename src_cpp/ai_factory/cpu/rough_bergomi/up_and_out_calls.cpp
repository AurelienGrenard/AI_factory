#include "ai_factory/cpu/rough_bergomi/up_and_out_calls.hpp"
#include "ai_factory/cpu/rough_bergomi/barrier_calls.hpp"

namespace ai_factory::cpu::rough_bergomi {

void price_up_and_out_call_batch(const cuda::RoughBergomiBarrierRow* rows, std::size_t row_count,
                       std::size_t num_paths, std::size_t num_steps,
                       cuda::MonteCarloOutput* outputs) {
    price_barrier_call_batch<true, false>(
        rows, row_count, num_paths, num_steps, outputs
    );
}

}  // namespace ai_factory::cpu::rough_bergomi
