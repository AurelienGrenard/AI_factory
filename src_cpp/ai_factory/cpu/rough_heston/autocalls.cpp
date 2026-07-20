#include "ai_factory/cpu/rough_heston/autocalls.hpp"

#include "ai_factory/cpu/rough_heston/common.hpp"
#include "ai_factory/cpu/common/philox.hpp"
#include "ai_factory/cpu/common/payoffs/autocall.hpp"
#include "ai_factory/cpu/common/payoffs/autocall_summary.hpp"

#include <vector>

namespace ai_factory::cpu::rough_heston {
namespace {

struct AutocallVisitorContext {
    const cuda::RoughHestonAutocallRow* row;
    std::vector<cuda::autocall_detail::PathState>* states;
    std::vector<cuda::autocall_detail::PathMetrics>* metrics;
};

bool observe_autocall(
    std::size_t path,
    std::size_t observation,
    double spot,
    void* raw_context
) {
    auto& context = *static_cast<AutocallVisitorContext*>(raw_context);
    const auto& row = *context.row;
    auto& state = (*context.states)[path];
    const double performance = spot / row.model.spot;
    const double time = row.model.maturity * static_cast<double>(observation)
                        / static_cast<double>(row.product.observation_count);
    const bool called = cuda::autocall_detail::observe(
        row.product,
        performance,
        observation,
        time,
        row.model.risk_free_rate,
        state
    );
    if (called || observation == row.product.observation_count) {
        (*context.metrics)[path] = cuda::autocall_detail::finish(
            row.product,
            performance,
            row.model.maturity,
            row.model.risk_free_rate,
            called ? observation : 0U,
            state
        );
    }
    return called;
}

}  // namespace

void price_autocall_batch(
    const cuda::RoughHestonAutocallRow* rows,
    std::size_t row_count,
    std::size_t num_paths,
    std::size_t num_steps,
    cuda::AutocallOutput* outputs
) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(row_count >= 4U)
#endif
    for (std::ptrdiff_t signed_index = 0;
         signed_index < static_cast<std::ptrdiff_t>(row_count);
         ++signed_index) {
        const auto index = static_cast<std::size_t>(signed_index);
        const auto& row = rows[index];
        const simulation::RoughHestonModel model{
            row.model.spot,
            row.model.risk_free_rate,
            row.model.dividend_yield,
            row.model.initial_variance,
            row.model.kappa,
            row.model.theta,
            row.model.volatility_of_variance,
            row.model.hurst,
            row.model.rho,
        };
        const simulation::TimeGrid grid{row.model.maturity, num_steps};
        const simulation::SimulationConfig simulation_config{
            row.model.seed,
            num_paths,
            simulation::kPhilox4x32_10BoxMuller,
        };
        std::vector<cuda::autocall_detail::PathState> states(num_paths);
        std::vector<cuda::autocall_detail::PathMetrics> metrics(num_paths);
        AutocallVisitorContext context{&row, &states, &metrics};
        simulation::visit_rough_heston_observation_spots(
            model,
            grid,
            simulation_config,
            row.product.observation_count,
            observe_autocall,
            &context
        );
        payoffs::AutocallSums sums{};
        for (const auto& path_metrics : metrics) {
            payoffs::add(sums, path_metrics);
        }
        outputs[index] = payoffs::summarize(sums, num_paths);
    }
}

}  // namespace ai_factory::cpu::rough_heston
