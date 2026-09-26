// Planner/native work arithmetic: compact tasks, empty selection, caps and overflow.
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
        require(defaults.tasks_per_price() == 10U
                && defaults.profile.threads_per_block == 256U,
            "Default MC gradient geometry changed.");
        const auto device_metadata =
            tuning::device_prepared_price_gradient_launch_metadata(
                defaults, 10U
            );
        require(
            device_metadata["profile_id"]
                    == "price_gradients_device_prepared_v2"
                && device_metadata["central_work_policy"]
                    == "sensitivity_zero_owns_price"
                && device_metadata["sensitivity_batches_per_price"] == 10U
                && device_metadata["maximum_live_scenarios"] == 3U
                && device_metadata["kernel_launches_per_price_batch"] == 1U
                && device_metadata["materialized_scenario_count"] == 0U
                && device_metadata["scenario_count"].is_null()
                && device_metadata["sensitivity_strategy"] == "mono"
                && device_metadata["sensitivity_kernel_variant"]
                    == (tuning::kPriceGradientLaunchBounds
                            && defaults.profile.threads_per_block
                                == tuning::kPriceGradientMonoThreadsPerBlock
                        ? "bounded"
                        : "unbounded")
                && device_metadata["sensitivity_launch_bounds"]["enabled"]
                    == tuning::kPriceGradientLaunchBounds
                && device_metadata["sensitivity_launch_bounds"]["mono"]
                        ["threads_per_block"]
                    == tuning::kPriceGradientMonoThreadsPerBlock
                && device_metadata["sensitivity_launch_bounds"]
                        ["node_evaluation"]["minimum_blocks_per_sm"]
                    == tuning::kPriceGradientNodeMinBlocksPerSm,
            "Device-prepared gradient metadata describes host scenarios."
        );
        const auto empty_device =
            tuning::device_prepared_price_gradient_launch_metadata(
                tuning::make_price_gradient_launch_plan(identity, 3U, 0U),
                0U
            );
        require(
            empty_device["central_work_policy"] == "price_only_delegation"
                && empty_device["maximum_live_scenarios"] == 1U
                && empty_device["represented_nodes_per_sensitivity"] == 0U,
            "Empty device-prepared selection metadata is inconsistent."
        );
        const tuning::PricingIdentity cev{tuning::PricingFamily::equity_step_mc,"cev","european_option",""};
        const auto cev_plan=tuning::make_price_gradient_launch_plan(cev,3,7);
        require(cev_plan.blocks_for(3)==21 && cev_plan.profile.threads_per_block==256,
            "CEV gradient planner lost the common geometry.");
        const tuning::PricingIdentity merton{tuning::PricingFamily::equity_exact_mc,"merton","european_option",""};
        const auto merton_plan=tuning::make_price_gradient_launch_plan(merton,3,7);
        require(merton_plan.blocks_for(3)==21 && merton_plan.profile.threads_per_block==256,
            "Merton gradient planner lost the common geometry.");
        const auto future_model = tuning::make_price_gradient_launch_plan(
            {tuning::PricingFamily::equity_step_mc,"future_model","future_product",""},3,17);
        require(future_model.blocks_for(3)==51,
            "The geometry planner still depends on a model capability list.");
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
                && american_plan.profile.threads_per_block
                    == tuning::kEarlyExerciseThreadsPerBlock
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
        for (const auto family : {
                 tuning::PricingFamily::gaussian_rate_lsm,
                 tuning::PricingFamily::terminal_forward_lsm,
             }) {
            const auto fixed_income =
                tuning::make_price_gradient_launch_plan(
                    {family, "rate_model", "bermudan_swaption", "curve"},
                    100U,
                    7U
                );
            require(
                fixed_income.profile.distribution
                        == tuning::PriceWorkDistribution::lsm
                    && fixed_income.profile.threads_per_block
                        == tuning::kEarlyExerciseThreadsPerBlock
                    && fixed_income.prices_per_launch == 16U
                    && fixed_income.blocks_for(16U)
                        == tuning::kFixedIncomeLsmBlocksPerPrice
                    && fixed_income.profile.evidence
                        == "docs/cuda/fixed-income-price-gradients-contract.md",
                "Fixed-income gradient planner lost native LSM geometry."
            );
            const auto fixed_income_metadata =
                tuning::device_prepared_price_gradient_launch_metadata(
                    fixed_income,
                    7U
                );
            require(
                fixed_income_metadata["central_work_policy"]
                        == "one central LSM policy plus frozen exercise replay"
                    && fixed_income_metadata["sensitivity_batch_size"] == 1U
                    && fixed_income_metadata["gradient_post_kernels_per_native_batch"]
                        == 2U
                    && fixed_income_metadata["materialized_scenario_count"] == 0U,
                "Fixed-income frozen-gradient metadata is inconsistent."
            );
        }
        for (std::size_t k=0; k<=10; ++k) {
            const auto plan = tuning::make_price_gradient_launch_plan(
                identity,3,k,4097
            );
            const auto tasks = k == 0 ? 1U : k;
            require(plan.blocks_for(3)==3*tasks,
                "Planner lost row/sensitivity parallelism.");
            require(plan.blocks_for(1)==tasks,
                "Warmup geometry differs from batch geometry.");
            const auto metadata = tuning::price_gradient_launch_metadata(plan,k);
            require(metadata["maximum_live_scenarios"]==(k == 0U ? 1U : 3U),
                "Incorrect live-state bound.");
            require(metadata["kernel_launches_per_price_batch"]==1U,
                "Compact sensitivities require more than one kernel launch.");
            const auto capped = tuning::make_price_gradient_launch_plan(
                identity,3,k,4097,{1024U,2U,2U}
            );
            require(capped.price_count_at(0)==2 && capped.price_count_at(2)==1,"Row memory cap changed.");
            require(capped.blocks_for(2)==2,"Grid cap ignored.");
        }
        const auto cf = tuning::make_price_gradient_launch_plan(
            {tuning::PricingFamily::closed_form,"black_scholes","european_option",""},1000,6,0);
        const auto metadata=tuning::price_gradient_launch_metadata(cf,6);
        require(metadata["sensitivity_batch_size"]==0 && metadata["sensitivity_batches_per_price"]==0
            && metadata["kernel_launches_per_price_batch"]==1,"CF incorrectly reports MC batches.");
        const auto device_cf_metadata =
            tuning::device_prepared_price_gradient_launch_metadata(cf, 6U);
        require(
            device_cf_metadata["sensitivity_batch_size"] == 0U
                && device_cf_metadata["sensitivity_batches_per_price"] == 0U
                && device_cf_metadata["work_distribution"]
                    == "one thread per row; sensitivities evaluated sequentially"
                && device_cf_metadata["materialized_scenario_count"] == 0U
                && !device_cf_metadata.contains(
                    "sensitivity_launch_bounds"
                ),
            "Device-prepared closed form reports MC block geometry."
        );
        const auto jamshidian = tuning::make_price_gradient_launch_plan(
            {tuning::PricingFamily::jamshidian,"cir","european_swaption",""},1000,7,0);
        const auto jamshidian_metadata =
            tuning::price_gradient_launch_metadata(jamshidian,7);
        require(jamshidian.profile.distribution == tuning::PriceWorkDistribution::block
            && jamshidian.tasks_per_price() == 1U
            && jamshidian_metadata["sensitivity_batch_size"] == 0U
            && jamshidian_metadata["central_work_policy"]
                == "sequential_sensitivities",
            "Jamshidian gradients lost cooperative closed-form planning.");
        const auto device_jamshidian =
            tuning::device_prepared_price_gradient_launch_metadata(
                jamshidian, 7U
            );
        require(
            device_jamshidian["central_work_policy"]
                    == "one_block_per_row_sequential_sensitivities"
                && device_jamshidian["sensitivity_batch_size"] == 0U
                && device_jamshidian["work_distribution"]
                    == "one block per row; sensitivities evaluated sequentially",
            "Compact Jamshidian metadata reports terminal-MC geometry."
        );
        const auto generic_jamshidian = tuning::make_price_gradient_launch_plan(
            {tuning::PricingFamily::jamshidian,"future_rate_model","future_swaption",""},3,12,0);
        require(generic_jamshidian.sensitivity_count == 12U,
            "Jamshidian geometry still depends on a model capability list.");
        bool overflow=false;
        try {
            (void)pg::sensitivity_task_count(
                std::numeric_limits<std::size_t>::max(),2U
            );
        }
        catch (const std::overflow_error&) { overflow=true; }
        require(overflow,"Task count overflow accepted.");
        std::cout << "Gradient execution-geometry contracts passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
