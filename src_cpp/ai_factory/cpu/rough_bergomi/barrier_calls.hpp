#pragma once

#include "ai_factory/cpu/common/payoffs/barrier.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/rough_bergomi/common.hpp"

namespace ai_factory::cpu::rough_bergomi {

template <bool Up, bool KnockIn>
void price_barrier_call_batch(
    const cuda::RoughBergomiBarrierRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::MonteCarloOutput* outputs
) {
    for (std::ptrdiff_t signed_index = 0;
         signed_index < static_cast<std::ptrdiff_t>(row_count);
         ++signed_index) {
        const auto index = static_cast<std::size_t>(signed_index);
        const auto& row = rows[index];
        const simulation::RoughBergomiModel model{
            row.model.spot,
            row.model.risk_free_rate,
            row.model.dividend_yield,
            row.model.forward_variance,
            row.model.eta,
            row.model.alpha,
            row.model.rho,
        };
        const auto paths = simulation::generate_rough_bergomi_spot_paths(
            model,
            {row.model.maturity, num_steps},
            {row.model.seed, num_paths, simulation::kPhilox4x32_10BoxMuller}
        );
        outputs[index] = payoffs::barrier_call_from_paths<Up, KnockIn>(
            paths,
            num_paths,
            num_steps,
            row.model.strike,
            row.product.barrier,
            std::exp(-row.model.risk_free_rate * row.model.maturity)
        );
    }
}

}  // namespace ai_factory::cpu::rough_bergomi
