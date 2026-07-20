#include "ai_factory/cpu/black_scholes/digital_calls.hpp"
#include "ai_factory/cpu/common/payoffs/terminal_calls.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/terminal_pricing.hpp"

#include <cmath>

namespace ai_factory::cpu::black_scholes {
void price_digital_call_batch(
    const cuda::BlackScholesRow* rows, std::size_t row_count,
    std::size_t num_paths, std::size_t, cuda::MonteCarloOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(row_count); ++i) {
        const auto& row = rows[i];
        auto terminals = simulation::philox_standard_normals(row.seed, num_paths, 0U);
        const double drift = (row.risk_free_rate - row.dividend_yield
                              - 0.5 * row.volatility * row.volatility) * row.maturity;
        const double diffusion = row.volatility * std::sqrt(row.maturity);
        for (double& value : terminals) value = row.spot * std::exp(drift + diffusion * value);
        outputs[i] = terminal_pricing::summarize(
            terminals, row.strike, std::exp(-row.risk_free_rate * row.maturity), payoffs::digital_call
        );
    }
}
}  // namespace ai_factory::cpu::black_scholes
