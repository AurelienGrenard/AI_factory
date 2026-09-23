// Host candidate geometry for selected gradients; native kernels validate compiled resources.
#pragma once
#include "tools/cuda/pricing_launch_plan.hpp"
#include "src/common/price_gradients/batching.hpp"

namespace ai_factory::workbench::offline::cuda_tuning {
namespace pg = ::ai_factory::workbench::price_gradients;
struct PriceGradientLaunchPlan : PricingLaunchPlan {
    std::size_t sensitivity_count;
    unsigned int sensitivity_batch_size;
    std::size_t batches_per_price() const {
        return identity.family == PricingFamily::closed_form ? 1U
            : pg::sensitivity_batch_count(sensitivity_count,sensitivity_batch_size);
    }
    std::size_t blocks_for(std::size_t rows) const {
        if (identity.family == PricingFamily::equity_lsm)
            return PricingLaunchPlan::blocks_for(rows);
        if (identity.family == PricingFamily::closed_form) return PricingLaunchPlan::blocks_for(rows);
        if (rows == 0U || rows > prices_per_launch)
            throw std::invalid_argument("Gradient price batch is outside the launch plan.");
        auto blocks = pg::gradient_task_count(rows,batches_per_price());
        if (profile.block_count_limit != 0U) blocks = std::min(blocks,profile.block_count_limit);
        return std::min(blocks,maximum_grid_x);
    }
};
inline PriceGradientLaunchPlan make_price_gradient_launch_plan(
    PricingIdentity identity, std::size_t rows, std::size_t sensitivities,
    std::size_t paths = kProductionPathsPerPrice, PricingLaunchLimits limits = {},
    unsigned int batch_size = pg::kDefaultSensitivityBatchSize
) {
    const bool american_heston = identity.product == "american_option"
        && identity.model == "heston"
        && identity.family == PricingFamily::equity_lsm;
    const bool european = identity.product == "european_option"
        && (identity.model == "black_scholes" || identity.model == "heston"
            || identity.model == "cev" || identity.model == "merton");
    if ((!european && !american_heston)
        || (identity.model != "black_scholes" && identity.model != "heston"
            && identity.model != "cev" && identity.model != "merton")
        || sensitivities > (american_heston ? 9U
            : identity.model == "black_scholes" ? 6U
            : identity.model == "heston" ? 10U : 7U)
        || (identity.model == "cev" && identity.family != PricingFamily::equity_step_mc)
        || (identity.model == "merton" && identity.family != PricingFamily::equity_exact_mc)
        || (identity.family != PricingFamily::closed_form
            && identity.family != PricingFamily::equity_exact_mc
            && identity.family != PricingFamily::equity_step_mc
            && identity.family != PricingFamily::equity_lsm))
        throw std::invalid_argument("Unsupported price-gradient launch identity or cardinality.");
    pg::validate_sensitivity_batch_size(batch_size);
    (void)pg::gradient_task_count(rows,pg::sensitivity_batch_count(sensitivities,batch_size));
    auto profile = pricing_profile(identity, rows);
    if (identity.family != PricingFamily::closed_form)
        profile.threads_per_block = pg::kDefaultThreadsPerBlock;
    if (identity.family == PricingFamily::equity_lsm) {
        if (batch_size != 1U)
            throw std::invalid_argument(
                "Frozen-exercise gradients require sensitivity batch size 1."
            );
        // Bound the durable checkpoint interval. The native LSM planner still
        // subdivides each invocation further according to free VRAM.
        profile.prices_per_launch = 16U;
    }
    profile.qualification = "gradient candidate; bounded resource checks, not performance-qualified";
    profile.evidence = "docs/cuda/equity-price-gradients-contract.md";
    return {make_pricing_launch_plan(identity, rows, paths, profile, limits), sensitivities,batch_size};
}
inline nlohmann::ordered_json price_gradient_launch_metadata(const PriceGradientLaunchPlan& plan, std::size_t sensitivities) {
    if (sensitivities != plan.sensitivity_count) throw std::invalid_argument("Gradient metadata selection mismatch.");
    auto result = pricing_launch_metadata(plan);
    const bool mc = plan.identity.family != PricingFamily::closed_form;
    const bool lsm = plan.identity.family == PricingFamily::equity_lsm;
    const auto width = plan.sensitivity_batch_size;
    const auto full = sensitivities/width, tail = sensitivities%width;
    result["price_moment_batches_per_price"] = 1U;
    result["central_work_policy"] = lsm ? "frozen_central_exercise_trace"
        : mc ? "host_classified_payoff_state_or_none" : "sequential_scenarios";
    result["sensitivity_batch_size"] = mc ? width : 0U;
    result["sensitivity_batches_per_price"] = mc ? plan.batches_per_price() : 0U;
    result["maximum_live_scenarios"] = lsm ? (sensitivities == 0U ? 1U : 2U)
        : mc ? 1U+2U*std::min(sensitivities,static_cast<std::size_t>(width)) : 1U;
    if (lsm) {
        result["kernel_launches_per_price_batch"] = nullptr;
        result["gradient_post_kernels_per_native_batch"] =
            sensitivities == 0U ? 0U : 2U;
        result["durable_prices_per_launch"] = plan.prices_per_launch;
        result["durable_launch_count"] = plan.price_launch_count();
    } else {
        result["kernel_launches_per_price_batch"] =
            mc && full != 0U && tail != 0U ? 2U : 1U;
    }
    result["block_count"] = plan.blocks_for(plan.prices_per_launch);
    if (mc) {
        const auto cap = plan.blocks_for(plan.prices_per_launch);
        if (lsm) {
            result["path_blocks_per_price"] = cap;
            result["gradient_tasks_per_durable_launch_upper_bound"] =
                plan.prices_per_launch * sensitivities;
        } else {
            result["full_batch_block_count"] = full == 0U ? 0U : std::min(cap,pg::gradient_task_count(plan.prices_per_launch,full));
            result["tail_batch_block_count"] = tail == 0U ? 0U : std::min(cap,plan.prices_per_launch);
        }
        result["work_distribution"] = lsm
            ? "central LSM batches plus grid (path shard, price*sensitivity)"
            : "one (price, sensitivity batch) per block at a time; exact-width tail grid";
    }
    result["measured_on"] = nullptr;
    result["profile_id"] = "price_gradients_single_price_candidate_v3";
    result["portability"] = "Unqualified gradient geometry; measure on the target GPU and toolchain.";
    result["sensitivity_count"] = sensitivities;
    result["scenario_count"] = 1U + 2U*sensitivities;
    result["gradient_layout"] = "row_major_selection_order";
    return result;
}
}  // namespace ai_factory::workbench::offline::cuda_tuning
