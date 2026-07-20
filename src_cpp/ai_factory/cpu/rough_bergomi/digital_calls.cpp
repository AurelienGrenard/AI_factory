#include "ai_factory/cpu/rough_bergomi/digital_calls.hpp"
#include "ai_factory/cpu/common/payoffs/terminal_calls.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/terminal_pricing.hpp"
#include "ai_factory/cpu/rough_bergomi/common.hpp"

#include <cmath>

namespace ai_factory::cpu::rough_bergomi {
void price_digital_call_batch(
    const cuda::RoughBergomiRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t num_steps, cuda::MonteCarloOutput* outputs
) {
    for (std::size_t i = 0; i < row_count; ++i) {
        const auto& row = rows[i];
        const simulation::RoughBergomiModel model{row.spot, row.risk_free_rate, row.dividend_yield, row.forward_variance, row.eta, row.alpha, row.rho};
        const auto terminals = simulation::generate_rough_bergomi_terminal_spots(
            model, {row.maturity, num_steps}, {row.seed, num_paths, simulation::kPhilox4x32_10BoxMuller}
        );
        outputs[i] = terminal_pricing::summarize(terminals, row.strike, std::exp(-row.risk_free_rate * row.maturity), payoffs::digital_call);
    }
}
}  // namespace ai_factory::cpu::rough_bergomi
