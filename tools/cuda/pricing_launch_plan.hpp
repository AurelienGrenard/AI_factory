// Pure host-side launch arithmetic; runtime engines own memory and kernel guards.
#pragma once

#include "tools/cuda/tuning_profile.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::offline::cuda_tuning {

inline std::size_t ceiling_divide(std::size_t count, std::size_t divisor) {
    if (divisor == 0U) throw std::invalid_argument("Launch divisor must be positive.");
    return count / divisor + (count % divisor != 0U);
}

struct PricingLaunchLimits {
    unsigned int maximum_threads_per_block = 1024U;
    std::size_t maximum_grid_x = 2147483647U;
    // Zero means unknown, not unlimited VRAM. LSM's native planner resolves it.
    std::size_t maximum_resident_prices = 0U;
};

struct PricingLaunchPlan {
    PricingIdentity identity;
    PricingProfile profile;
    std::size_t price_count;
    std::size_t paths_per_price;
    std::size_t prices_per_launch;
    std::size_t maximum_grid_x;
    bool memory_batching_resolved;

    std::size_t price_launch_count() const {
        return ceiling_divide(price_count, prices_per_launch);
    }

    std::size_t price_count_at(std::size_t offset) const {
        if (offset >= price_count) throw std::out_of_range("Price batch offset exceeds the plan.");
        return std::min(prices_per_launch, price_count - offset);
    }

    std::size_t blocks_for(std::size_t batch_count) const {
        if (batch_count == 0U || batch_count > prices_per_launch)
            throw std::invalid_argument("Price batch is outside the launch plan.");
        if (profile.distribution == PriceWorkDistribution::lsm)
            return profile.blocks_per_price;  // grid.x; native planner owns grid.y.
        if (profile.distribution == PriceWorkDistribution::fft)
            throw std::logic_error("FFT grids belong to the compiled FFT specialization.");
        std::size_t blocks = profile.distribution == PriceWorkDistribution::thread
            ? ceiling_divide(batch_count, profile.threads_per_block) : batch_count;
        if (profile.block_count_limit != 0U)
            blocks = std::min(blocks, profile.block_count_limit);
        return std::min(blocks, maximum_grid_x);
    }
};

inline PricingLaunchPlan make_pricing_launch_plan(
    PricingIdentity identity,
    std::size_t price_count,
    std::size_t paths_per_price,
    PricingProfile profile,
    PricingLaunchLimits limits = {}
) {
    if (price_count == 0U) throw std::invalid_argument("A pricing plan requires prices.");
    (void)pricing_family_name(identity.family);
    switch (profile.distribution) {
    case PriceWorkDistribution::thread:
    case PriceWorkDistribution::block:
    case PriceWorkDistribution::lsm:
    case PriceWorkDistribution::fft:
        break;
    default:
        throw std::invalid_argument("Unknown pricing work distribution.");
    }
    const bool analytical = identity.family == PricingFamily::closed_form
        || identity.family == PricingFamily::jamshidian;
    if ((analytical && paths_per_price != 0U) || (!analytical && paths_per_price < 2U))
        throw std::invalid_argument("Closed forms require zero paths; MC requires at least two.");
    if (profile.threads_per_block == 0U || profile.threads_per_block % 32U != 0U
        || profile.threads_per_block > limits.maximum_threads_per_block)
        throw std::invalid_argument("Pricing block size must be an admissible whole number of warps.");
    if (limits.maximum_grid_x == 0U
        || limits.maximum_grid_x > std::numeric_limits<unsigned int>::max())
        throw std::invalid_argument("Pricing grid limit is not representable by dim3.");
    if (profile.distribution == PriceWorkDistribution::lsm
        && (profile.blocks_per_price == 0U || profile.blocks_per_price > limits.maximum_grid_x))
        throw std::invalid_argument("LSM blocks per price exceed the launch limits.");
    if (profile.distribution == PriceWorkDistribution::fft && profile.path_chunk_size == 0U)
        throw std::invalid_argument("FFT pricing requires a positive path chunk size.");
    std::size_t batch = profile.prices_per_launch == 0U
        ? price_count : std::min(price_count, profile.prices_per_launch);
    if (limits.maximum_resident_prices != 0U)
        batch = std::min(batch, limits.maximum_resident_prices);
    return {identity, profile, price_count, paths_per_price, batch, limits.maximum_grid_x,
        profile.distribution != PriceWorkDistribution::lsm || limits.maximum_resident_prices != 0U};
}

inline PricingLaunchPlan make_pricing_launch_plan(
    PricingIdentity identity,
    std::size_t price_count,
    std::size_t paths_per_price = kProductionPathsPerPrice,
    PricingLaunchLimits limits = {}
) {
    return make_pricing_launch_plan(identity, price_count, paths_per_price,
        pricing_profile(identity, price_count), limits);
}

inline PricingLaunchPlan make_equity_price_delta_launch_plan(
    PricingIdentity identity, std::size_t price_count,
    std::size_t paths_per_price = kProductionPathsPerPrice,
    PricingLaunchLimits limits = {}
) {
    auto profile = pricing_profile(identity, price_count);
    if (identity.family == PricingFamily::equity_exact_mc
        || identity.family == PricingFamily::equity_step_mc
        || identity.family == PricingFamily::rough_n_factor) {
        // Three live payoff states can exceed 128 registers/thread. At 512
        // threads this exhausts a 64K-register block (observed for Bates).
        // A conservative candidate, not an occupancy/performance optimum.
        profile.threads_per_block = std::min(profile.threads_per_block, 256U);
    } else if (identity.family != PricingFamily::closed_form
               && identity.family != PricingFamily::equity_lsm
               && identity.family != PricingFamily::rough_fft) {
        throw std::invalid_argument("Unsupported equity price-delta launch family.");
    }
    profile.qualification = "inherited candidate; MC capped at 256 threads; not delta-tuned";
    profile.evidence = "docs/cuda/equity-price-delta-contract.md";
    return make_pricing_launch_plan(identity, price_count, paths_per_price, profile, limits);
}

inline nlohmann::ordered_json pricing_launch_metadata(const PricingLaunchPlan& plan) {
    auto result = metadata(pricing_family_name(plan.identity.family));
    result["model"] = plan.identity.model;
    result["product"] = plan.identity.product;
    result["curve"] = plan.identity.curve;
    result["selection"] = "host launch plan; no runtime autotuning";
    result["qualification"] = plan.profile.qualification;
    result["evidence"] = plan.profile.evidence;
    result["price_count"] = plan.price_count;
    result["paths_per_price"] = plan.paths_per_price;
    result["threads_per_block"] = plan.profile.threads_per_block;
    result["prices_per_launch"] = plan.memory_batching_resolved
        ? nlohmann::ordered_json(plan.prices_per_launch) : nlohmann::ordered_json(nullptr);
    result["price_launch_count"] = plan.memory_batching_resolved
        ? nlohmann::ordered_json(plan.price_launch_count()) : nlohmann::ordered_json(nullptr);
    result["memory_batching"] = plan.memory_batching_resolved
        ? "price launch bound; engine workspace guards still apply" : "native LSM VRAM planner; unresolved offline";
    switch (plan.profile.distribution) {
    case PriceWorkDistribution::thread:
    case PriceWorkDistribution::block:
        result["work_distribution"] = plan.profile.distribution == PriceWorkDistribution::thread
            ? "one price per thread at a time" : "one price per block at a time";
        result["block_count"] = plan.blocks_for(plan.prices_per_launch);
        break;
    case PriceWorkDistribution::lsm:
        result["work_distribution"] = "multiple path blocks per price; native multi-phase LSM";
        result["blocks_per_price"] = plan.profile.blocks_per_price;
        break;
    case PriceWorkDistribution::fft:
        result.erase("threads_per_block");
        result["pricing_path_threads"] = plan.profile.threads_per_block;
        result["pricing_finalization_threads"] = kVolterraPricingFinalizationThreads;
        result["fft_geometry_owner"] = "src/common/volterra/hybrid_fft_tuning.cuh";
        result["work_distribution"] = "compiled FFT specialization; sequential price submissions";
        result["path_chunk_size"] = plan.profile.path_chunk_size;
        result["chunks_per_price"] = ceiling_divide(plan.paths_per_price, plan.profile.path_chunk_size);
        break;
    }
    return result;
}

}  // namespace ai_factory::workbench::offline::cuda_tuning
