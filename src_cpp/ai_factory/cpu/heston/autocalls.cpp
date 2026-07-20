#include "ai_factory/cpu/heston/autocalls.hpp"

#include "ai_factory/cpu/heston/common.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/payoffs/autocall.hpp"
#include "ai_factory/cpu/common/payoffs/autocall_summary.hpp"

namespace ai_factory::cpu::heston {

void price_autocall_batch(
    const cuda::HestonAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::AutocallOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t signed_index = 0;
         signed_index < static_cast<std::ptrdiff_t>(row_count);
         ++signed_index) {
        const auto index = static_cast<std::size_t>(signed_index);
        const auto& row = rows[index];
        const simulation::HestonModel model{
            row.model.spot,
            row.model.risk_free_rate,
            row.model.dividend_yield,
            row.model.initial_variance,
            row.model.kappa,
            row.model.theta,
            row.model.volatility_of_variance,
            row.model.rho,
        };
        const simulation::TimeGrid grid{row.model.maturity, num_steps};
        const simulation::SimulationConfig simulation_config{
            row.model.seed,
            num_paths,
            simulation::kPhilox4x32_10BoxMuller,
        };
        const auto observations = simulation::generate_heston_observation_spots(
            model,
            grid,
            simulation_config,
            row.product.observation_count,
            static_cast<simulation::HestonSimulationScheme>(row.model.scheme)
        );
        const auto observation_count = row.product.observation_count;
        payoffs::AutocallSums sums{};
        for (std::size_t path = 0; path < num_paths; ++path) {
            payoffs::add(
                sums,
                payoffs::autocall_from_observations(
                    observations.data() + path * observation_count,
                    row.model.spot,
                    row.model.maturity,
                    row.model.risk_free_rate,
                    row.product
                )
            );
        }
        outputs[index] = payoffs::summarize(sums, num_paths);
    }
}

}  // namespace ai_factory::cpu::heston
