// Host-only geometry, overflow and native-workspace ownership regression tests.
#include "tools/cuda/pricing_launch_plan.hpp"

#include <iostream>
#include <limits>

namespace tuning = ai_factory::workbench::offline::cuda_tuning;

void require(bool condition) {
    if (!condition) throw std::runtime_error("Pricing launch plan assertion failed.");
}

template<class Function>
void rejects(Function&& function) {
    try { function(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Expected invalid pricing plan to be rejected.");
}

int main() {
    try {
        const tuning::PricingIdentity mc{tuning::PricingFamily::equity_step_mc, "heston", "european_option", ""};
        for (const auto count : {1U, 1000U, 4096U, 4097U, 1000000U, 1048576U}) {
            const auto plan = tuning::make_pricing_launch_plan(mc, count);
            require(plan.paths_per_price == 1048576U);
            std::size_t covered = 0U;
            while (covered < count) {
                const auto batch = plan.price_count_at(covered);
                require(batch > 0U && plan.blocks_for(batch) <= batch);
                covered += batch;
            }
            require(covered == count);
        }
        auto profile = tuning::pricing_profile(mc);
        profile.prices_per_launch = 0U;
        profile.block_count_limit = 4096U;
        auto plan = tuning::make_pricing_launch_plan(mc, 1048576U, 1048576U, profile);
        require(plan.blocks_for(plan.prices_per_launch) == 4096U);
        require(plan.price_launch_count() == 1U);
        const tuning::PricingIdentity analytical{tuning::PricingFamily::closed_form, "black_scholes", "european_option", ""};
        plan = tuning::make_pricing_launch_plan(analytical, 1000U, 0U);
        require(plan.blocks_for(1000U) == tuning::ceiling_divide(1000U, tuning::kAnalyticalThreadsPerBlock));
        const auto large = std::numeric_limits<std::size_t>::max();
        plan = tuning::make_pricing_launch_plan(analytical, large, 0U);
        require(plan.blocks_for(large) <= plan.maximum_grid_x);
        require(tuning::ceiling_divide(large, 2U) == large / 2U + 1U);
        const tuning::PricingIdentity lsm{tuning::PricingFamily::equity_lsm, "heston", "american_option", ""};
        plan = tuning::make_pricing_launch_plan(lsm, 1000000U);
        require(!plan.memory_batching_resolved);
        require(tuning::pricing_launch_metadata(plan)["price_launch_count"].is_null());
        plan = tuning::make_pricing_launch_plan(lsm, 1000U, 1048576U, {1024U, 2147483647U, 7U});
        require(plan.memory_batching_resolved && plan.prices_per_launch == 7U);
        require(plan.price_count_at(994U) == 6U && plan.paths_per_price == 1048576U);
        const tuning::PricingIdentity fft{tuning::PricingFamily::rough_fft, "rough_bergomi", "european_option", ""};
        const auto paired = tuning::make_equity_price_delta_launch_plan(mc, 1000U);
        require(paired.paths_per_price == 1048576U && paired.profile.threads_per_block <= 256U);
        require(tuning::make_equity_price_delta_launch_plan(lsm, 1000U).profile.blocks_per_price
                == tuning::pricing_profile(lsm).blocks_per_price);
        require(tuning::make_equity_price_delta_launch_plan(analytical, 1000U, 0U).profile.threads_per_block
                == tuning::pricing_profile(analytical).threads_per_block);
        const auto fft_delta = tuning::make_equity_price_delta_launch_plan(fft, 1000U);
        require(fft_delta.profile.path_chunk_size == tuning::pricing_profile(fft, 1000U).path_chunk_size);
        require(fft_delta.paths_per_price == tuning::kProductionPathsPerPrice);
        plan = tuning::make_pricing_launch_plan(fft, 1000U);
        require(plan.profile.path_chunk_size == tuning::kVolterraPathChunkSize);
        require(plan.paths_per_price == 1048576U && plan.prices_per_launch == 1U);
        rejects([&] { plan.blocks_for(1U); });
        rejects([&] { tuning::make_pricing_launch_plan(mc, 0U); });
        rejects([&] { tuning::make_pricing_launch_plan(mc, 1U, 1U); });
        rejects([&] { tuning::make_pricing_launch_plan(analytical, 1U, 1048576U); });
        rejects([&] { tuning::make_pricing_launch_plan(mc, 1000U, 1048576U, {32U}); });
        rejects([&] { tuning::ceiling_divide(1U, 0U); });
        profile.distribution = static_cast<tuning::PriceWorkDistribution>(-1);
        rejects([&] { tuning::make_pricing_launch_plan(mc, 1000U, 1048576U, profile); });
        const tuning::PricingIdentity cir{
            tuning::PricingFamily::jamshidian, "cir", "european_swaption", ""};
        const auto cir_plan = tuning::make_pricing_launch_plan(cir, 1000U, 0U);
        require(cir_plan.profile.threads_per_block == 128U);
        require(cir_plan.profile.qualification == "native_geometry_pending_confirmation");
        if (tuning::kProfileId == "sm89_reference_v1") {
            const tuning::PricingIdentity fitted{
                tuning::PricingFamily::jamshidian, "cir_plus_plus", "european_swaption", "svensson"};
            const auto catalogue = tuning::make_pricing_launch_plan(fitted, 1000U, 0U);
            require(catalogue.profile.threads_per_block == 256U);
            require(catalogue.blocks_for(1000U) == 256U);
            const auto large_batch = tuning::make_pricing_launch_plan(fitted, 1048576U, 0U);
            require(large_batch.profile.threads_per_block == 64U);
            require(large_batch.blocks_for(1048576U) == 1048576U);
        }
        std::cout << "Pricing launch plan tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
