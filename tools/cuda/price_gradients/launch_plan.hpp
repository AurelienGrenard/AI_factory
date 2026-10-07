// Host candidate geometry for selected gradients; native kernels validate compiled resources.
#pragma once
#include "tools/cuda/pricing_launch_plan.hpp"
#include "src/common/price_gradients/execution_geometry.hpp"
#include "src/common/price_gradients/sensitivity_graph_plan.hpp"

namespace ai_factory::workbench::offline::cuda_tuning {
namespace pg = ::ai_factory::workbench::price_gradients;

inline constexpr bool is_lsm_family(PricingFamily family) {
    return family == PricingFamily::equity_lsm
        || family == PricingFamily::gaussian_rate_lsm
        || family == PricingFamily::terminal_forward_lsm;
}

struct PriceGradientLaunchPlan : PricingLaunchPlan {
    std::size_t sensitivity_count;
    std::size_t tasks_per_price() const {
        return (identity.family == PricingFamily::closed_form
                || identity.family == PricingFamily::jamshidian) ? 1U
            : sensitivity_count == 0U ? 1U : sensitivity_count;
    }
    std::size_t blocks_for(std::size_t rows) const {
        if (is_lsm_family(identity.family))
            return PricingLaunchPlan::blocks_for(rows);
        if (identity.family == PricingFamily::closed_form
            || identity.family == PricingFamily::jamshidian)
            return PricingLaunchPlan::blocks_for(rows);
        if (rows == 0U || rows > prices_per_launch)
            throw std::invalid_argument("Gradient price batch is outside the launch plan.");
        auto blocks = pg::sensitivity_task_count(rows, sensitivity_count);
        if (profile.block_count_limit != 0U) blocks = std::min(blocks,profile.block_count_limit);
        return std::min(blocks,maximum_grid_x);
    }
};
inline PriceGradientLaunchPlan make_price_gradient_launch_plan(
    PricingIdentity identity, std::size_t rows, std::size_t sensitivities,
    std::size_t paths = kProductionPathsPerPrice,
    PricingLaunchLimits limits = {}
) {
    // Capability and selected-coordinate validation belong to the composed
    // scenario builder. This planner owns geometry only and never enumerates
    // models, products or parameter cardinalities.
    if (identity.family != PricingFamily::closed_form
            && identity.family != PricingFamily::equity_exact_mc
            && identity.family != PricingFamily::equity_step_mc
            && identity.family != PricingFamily::fixed_income_mc
            && !is_lsm_family(identity.family)
            && identity.family != PricingFamily::jamshidian)
        throw std::invalid_argument("Unsupported price-gradient execution family.");
    (void)pg::sensitivity_task_count(rows, sensitivities);
    auto profile = pricing_profile(identity, rows);
    if (identity.family != PricingFamily::closed_form
        && identity.family != PricingFamily::jamshidian
        && !is_lsm_family(identity.family)) {
        profile.threads_per_block = pg::kDefaultThreadsPerBlock;
    }
    if (is_lsm_family(identity.family)) {
        // Bound the durable checkpoint interval. The native LSM planner still
        // subdivides each invocation further according to free VRAM.
        profile.prices_per_launch = 16U;
    }
    profile.qualification = "gradient candidate; bounded resource checks, not performance-qualified";
    profile.evidence = identity.family == PricingFamily::gaussian_rate_lsm
            || identity.family == PricingFamily::terminal_forward_lsm
        ? "docs/cuda/fixed-income-price-gradients-contract.md"
        : "docs/cuda/equity-price-gradients-contract.md";
    return {
        make_pricing_launch_plan(identity, rows, paths, profile, limits),
        sensitivities
    };
}
inline nlohmann::ordered_json price_gradient_launch_metadata(const PriceGradientLaunchPlan& plan, std::size_t sensitivities) {
    if (sensitivities != plan.sensitivity_count) throw std::invalid_argument("Gradient metadata selection mismatch.");
    auto result = pricing_launch_metadata(plan);
    const bool mc = plan.identity.family != PricingFamily::closed_form
        && plan.identity.family != PricingFamily::jamshidian;
    const bool lsm = is_lsm_family(plan.identity.family);
    result["price_moment_batches_per_price"] = 1U;
    result["central_work_policy"] = lsm ? "frozen_central_exercise_trace"
        : mc ? "sensitivity_zero_owns_price" : "sequential_sensitivities";
    result["sensitivity_batch_size"] =
        mc && sensitivities != 0U ? pg::kSensitivitiesPerBlock : 0U;
    result["sensitivity_batches_per_price"] =
        mc && sensitivities != 0U ? sensitivities : 0U;
    result["maximum_live_scenarios"] = lsm ? (sensitivities == 0U ? 1U : 2U)
        : mc ? (sensitivities == 0U ? 1U : 3U) : 1U;
    if (lsm) {
        result["kernel_launches_per_price_batch"] = nullptr;
        result["gradient_post_kernels_per_native_batch"] =
            sensitivities == 0U ? 0U : 2U;
        result["durable_prices_per_launch"] = plan.prices_per_launch;
        result["durable_launch_count"] = plan.price_launch_count();
    } else {
        result["kernel_launches_per_price_batch"] = 1U;
    }
    result["block_count"] = plan.blocks_for(plan.prices_per_launch);
    if (mc) {
        const auto cap = plan.blocks_for(plan.prices_per_launch);
        if (lsm) {
            result["path_blocks_per_price"] = cap;
            result["gradient_tasks_per_durable_launch_upper_bound"] =
                plan.prices_per_launch * sensitivities;
        } else {
            result["sensitivity_block_count"] =
                sensitivities == 0U ? 0U : cap;
        }
        result["work_distribution"] = lsm
            ? "central LSM batches plus grid (path shard, price*sensitivity)"
            : "grid (row, sensitivity); one sensitivity per block";
    }
    result["measured_on"] = nullptr;
    result["profile_id"] = "price_gradients_single_price_candidate_v3";
    result["portability"] = "Unqualified gradient geometry; measure on the target GPU and toolchain.";
    result["sensitivity_count"] = sensitivities;
    result["scenario_count"] = nullptr;
    result["materialized_scenario_count"] = 0U;
    result["gradient_layout"] = "row_major_selection_order";
    return result;
}

inline nlohmann::ordered_json device_prepared_price_gradient_launch_metadata(
    const PriceGradientLaunchPlan& plan,
    std::size_t sensitivities
) {
    auto result = price_gradient_launch_metadata(plan, sensitivities);
    const bool has_sensitivities = sensitivities != 0U;
    const bool closed_form =
        plan.identity.family == PricingFamily::closed_form;
    const bool cooperative =
        plan.identity.family == PricingFamily::jamshidian;
    const bool lsm = is_lsm_family(plan.identity.family);
    const bool analytical = closed_form || cooperative;
    if (!analytical && !lsm) {
        result["sensitivity_strategy"] = "mono";
        result["sensitivity_launch_bounds"] =
            price_gradient_launch_bounds_metadata();
        result["sensitivity_kernel_variant"] =
            kPriceGradientLaunchBounds
                && plan.profile.threads_per_block
                    == kPriceGradientMonoThreadsPerBlock
            ? "bounded"
            : "unbounded";
    }
    if (lsm) {
        result["central_work_policy"] =
            "one central LSM policy plus frozen exercise replay";
        result["sensitivity_batch_size"] =
            has_sensitivities ? 1U : 0U;
        result["sensitivity_batches_per_price"] =
            has_sensitivities ? sensitivities : 0U;
        result["maximum_live_scenarios"] =
            has_sensitivities ? 3U : 1U;
        result["kernel_launches_per_price_batch"] = nullptr;
        result["gradient_post_kernels_per_native_batch"] =
            has_sensitivities ? 2U : 0U;
        result["work_distribution"] = has_sensitivities
            ? "central LSM plus grid (path shard, price*sensitivity)"
            : "price-only LSM delegation";
        result["profile_id"] = "price_gradients_device_prepared_lsm_v1";
        result["represented_nodes_per_sensitivity"] =
            has_sensitivities ? 3U : 0U;
        return result;
    }
    result["central_work_policy"] = closed_form
        ? "one_thread_per_row_sequential_sensitivities"
        : cooperative
            ? "one_block_per_row_sequential_sensitivities"
        : has_sensitivities
            ? "sensitivity_zero_owns_price"
            : "price_only_delegation";
    result["sensitivity_batch_size"] =
        analytical || !has_sensitivities ? 0U : 1U;
    result["sensitivity_batches_per_price"] =
        analytical || !has_sensitivities ? 0U : sensitivities;
    result["maximum_live_scenarios"] = has_sensitivities ? 3U : 1U;
    result["kernel_launches_per_price_batch"] = 1U;
    result["work_distribution"] = closed_form
        ? "one thread per row; sensitivities evaluated sequentially"
        : cooperative
            ? "one block per row; sensitivities evaluated sequentially"
        : has_sensitivities
            ? "grid (row, sensitivity); block y=0 owns the central price"
            : "price-only kernel";
    result["profile_id"] = "price_gradients_device_prepared_v2";
    result["represented_nodes_per_sensitivity"] =
        has_sensitivities ? 3U : 0U;
    return result;
}

inline nlohmann::ordered_json diagonal_node_graph_launch_metadata(
    const PriceGradientLaunchPlan& plan,
    std::size_t sensitivities
) {
    const auto graph = pg::make_sensitivity_graph_plan(
        pg::SensitivityRequest{pg::SensitivityOrders::first_and_second},
        sensitivities
    );
    auto result = device_prepared_price_gradient_launch_metadata(
        plan, sensitivities
    );
    for (const char* field : {
             "block_count",
             "full_batch_block_count",
             "tail_batch_block_count",
             "sensitivity_batch_size",
             "sensitivity_batches_per_price",
             "sensitivity_block_count",
             "sensitivity_kernel_variant",
             "price_moment_batches_per_price",
             "path_blocks_per_price",
             "gradient_tasks_per_durable_launch_upper_bound",
             "gradient_post_kernels_per_native_batch",
         }) {
        result.erase(field);
    }
    result["kernel_launches_per_price_batch"] = nullptr;
    result["central_work_policy"] =
        "central node zero evaluated once per row and path chunk";
    result["work_distribution"] =
        "price and path chunks; unique diagonal nodes evaluated before reconstruction";
    result["profile_id"] = "price_gradients_diagonal_node_graph_v1";
    result["sensitivity_strategy"] = "node_graph";
    result["maximum_live_scenarios"] = graph.node_capacity;
    result["sensitivity_graph_node_capacity"] = graph.node_capacity;
    result["represented_nodes_per_sensitivity"] =
        sensitivities == 0U ? 0U : 4U;
    return result;
}

inline nlohmann::ordered_json mixed_node_graph_launch_metadata(
    const PriceGradientLaunchPlan& plan,
    std::size_t sensitivities,
    const pg::SensitivityGraphPlan& graph
) {
    if (sensitivities != plan.sensitivity_count
        || graph.coordinate_uses.size() != sensitivities) {
        throw std::invalid_argument(
            "Mixed node-graph metadata selection mismatch."
        );
    }
    if (graph.mixed_second.empty()) {
        throw std::invalid_argument(
            "Mixed node-graph metadata requires a mixed derivative."
        );
    }

    auto result = device_prepared_price_gradient_launch_metadata(
        plan, sensitivities
    );
    for (const char* field : {
             "block_count",
             "full_batch_block_count",
             "tail_batch_block_count",
             "sensitivity_batch_size",
             "sensitivity_batches_per_price",
             "sensitivity_block_count",
             "sensitivity_kernel_variant",
             "price_moment_batches_per_price",
             "path_blocks_per_price",
             "gradient_tasks_per_durable_launch_upper_bound",
             "gradient_post_kernels_per_native_batch",
         }) {
        result.erase(field);
    }

    const bool closed_form =
        plan.identity.family == PricingFamily::closed_form;
    const bool cooperative =
        plan.identity.family == PricingFamily::jamshidian;
    const bool lsm = is_lsm_family(plan.identity.family);
    result["kernel_launches_per_price_batch"] =
        closed_form || cooperative
            ? nlohmann::ordered_json(1U)
            : nlohmann::ordered_json(nullptr);
    result["central_work_policy"] = lsm
        ? "one central exercise policy reused by every frozen graph node"
        : "central node zero evaluated once per row and path chunk";
    result["work_distribution"] = closed_form
        ? "one thread per row; selected graph nodes evaluated sequentially"
        : cooperative
            ? "one block per row; selected graph nodes evaluated cooperatively"
        : lsm
            ? "central LSM batches followed by frozen node-graph replay and reconstruction"
            : "price and path chunks; graph nodes evaluated before derivative reconstruction";
    result["profile_id"] = "price_gradients_mixed_node_graph_v1";
    result["sensitivity_strategy"] = "mixed_node_graph";
    result["sensitivity_count"] = sensitivities;
    result["scenario_count"] = nullptr;
    result["materialized_scenario_count"] = 0U;
    result["gradient_layout"] = "row_major_selected_sensitivity_graph";
    result["sensitivity_graph_node_capacity"] = graph.node_capacity;
    result["first_sensitivity_count"] = graph.first.size();
    result["diagonal_hessian_count"] = graph.diagonal_second.size();
    result["mixed_hessian_count"] = graph.mixed_second.size();
    result["maximum_live_scenarios"] = graph.node_capacity;
    result["represented_nodes_per_sensitivity"] = 4U;
    result["mixed_corner_nodes_per_pair"] = 4U;
    result["requested_orders"] = nlohmann::ordered_json::array();
    if (!graph.first.empty()) {
        result["requested_orders"].push_back("first");
    }
    if (!graph.diagonal_second.empty()) {
        result["requested_orders"].push_back("diagonal_second");
    }
    result["requested_orders"].push_back("mixed_second");
    return result;
}
}  // namespace ai_factory::workbench::offline::cuda_tuning
