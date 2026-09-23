// Planner/native work arithmetic: tails, empty selection, grid caps and overflow.
#include "tools/cuda/price_gradients/launch_plan.hpp"
#include <iostream>

namespace tuning = ai_factory::workbench::offline::cuda_tuning;
namespace pg = ai_factory::workbench::price_gradients;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main() {
    try {
        const tuning::PricingIdentity identity{tuning::PricingFamily::equity_step_mc,"heston","european_option",""};
        const auto defaults = tuning::make_price_gradient_launch_plan(identity,3,10);
        require(defaults.sensitivity_batch_size == 1U && defaults.profile.threads_per_block == 256U,
            "Default MC gradient geometry changed.");
        const tuning::PricingIdentity cev{tuning::PricingFamily::equity_step_mc,"cev","european_option",""};
        const auto cev_plan=tuning::make_price_gradient_launch_plan(cev,3,7);
        require(cev_plan.blocks_for(3)==21 && cev_plan.profile.threads_per_block==256,
            "CEV gradient planner lost the common geometry.");
        bool excess=false;
        try { (void)tuning::make_price_gradient_launch_plan(cev,3,8); }
        catch (const std::invalid_argument&) { excess=true; }
        require(excess,"CEV accepted more than seven coordinates.");
        const tuning::PricingIdentity merton{tuning::PricingFamily::equity_exact_mc,"merton","european_option",""};
        const auto merton_plan=tuning::make_price_gradient_launch_plan(merton,3,7);
        require(merton_plan.blocks_for(3)==21 && merton_plan.profile.threads_per_block==256,
            "Merton gradient planner lost the common geometry.");
        excess=false;
        try { (void)tuning::make_price_gradient_launch_plan(merton,3,8); }
        catch (const std::invalid_argument&) { excess=true; }
        require(excess,"Merton accepted more than seven coordinates.");
        const tuning::PricingIdentity american{
            tuning::PricingFamily::equity_lsm,
            "heston",
            "american_option",
            ""
        };
        const auto american_plan = tuning::make_price_gradient_launch_plan(
            american, 100U, 9U
        );
        require(american_plan.profile.distribution
                    == tuning::PriceWorkDistribution::lsm
                && american_plan.profile.threads_per_block == 256U
                && american_plan.prices_per_launch == 16U
                && american_plan.blocks_for(16U)
                    == tuning::kEarlyExerciseBlocksPerPrice,
                "American gradient planner lost LSM geometry/checkpoint batching.");
        const auto american_metadata =
            tuning::price_gradient_launch_metadata(american_plan, 9U);
        require(american_metadata["central_work_policy"]
                    == "frozen_central_exercise_trace"
                && american_metadata["gradient_post_kernels_per_native_batch"]
                    == 2U
                && american_metadata["maximum_live_scenarios"] == 2U,
                "American frozen-gradient metadata is inconsistent.");
        excess = false;
        try {
            (void)tuning::make_price_gradient_launch_plan(
                american, 3U, 10U
            );
        } catch (const std::invalid_argument&) { excess = true; }
        require(excess, "American Heston accepted more than nine coordinates.");
        excess = false;
        try {
            (void)tuning::make_price_gradient_launch_plan(
                american, 3U, 4U, 4097U, {}, 2U
            );
        } catch (const std::invalid_argument&) { excess = true; }
        require(excess, "American frozen exercise accepted B greater than one.");
        for (unsigned width : {1U,2U,4U}) for (std::size_t k=0; k<=10; ++k) {
            const auto plan = tuning::make_price_gradient_launch_plan(identity,3,k,4097,{},width);
            const auto batches = k == 0 ? 1U : (k+width-1U)/width;
            require(plan.blocks_for(3)==3*batches,"Planner lost row/batch parallelism.");
            require(plan.blocks_for(1)==batches,"Warmup geometry differs from batch geometry.");
            const auto metadata = tuning::price_gradient_launch_metadata(plan,k);
            require(metadata["maximum_live_scenarios"]==1+2*std::min(k,std::size_t(width)),"Incorrect live-state bound.");
            require(metadata["kernel_launches_per_price_batch"]==((k/width && k%width)?2:1),"Incorrect tail launches.");
            const auto capped = tuning::make_price_gradient_launch_plan(identity,3,k,4097,{1024U,2U,2U},width);
            require(capped.price_count_at(0)==2 && capped.price_count_at(2)==1,"Row memory cap changed.");
            require(capped.blocks_for(2)==2,"Grid cap ignored.");
        }
        const auto cf = tuning::make_price_gradient_launch_plan(
            {tuning::PricingFamily::closed_form,"black_scholes","european_option",""},1000,6,0,{},4);
        const auto metadata=tuning::price_gradient_launch_metadata(cf,6);
        require(metadata["sensitivity_batch_size"]==0 && metadata["sensitivity_batches_per_price"]==0
            && metadata["kernel_launches_per_price_batch"]==1,"CF incorrectly reports MC batches.");
        for (unsigned width : {0U,3U,5U}) {
            bool rejected=false;
            try { (void)tuning::make_price_gradient_launch_plan(identity,3,10,4097,{},width); }
            catch (const std::invalid_argument&) { rejected=true; }
            require(rejected,"Unsupported width accepted.");
        }
        bool overflow=false;
        try { (void)pg::gradient_task_count(std::numeric_limits<std::size_t>::max(),2U); }
        catch (const std::overflow_error&) { overflow=true; }
        require(overflow,"Task count overflow accepted.");
        std::cout << "Gradient batching planner contracts passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
