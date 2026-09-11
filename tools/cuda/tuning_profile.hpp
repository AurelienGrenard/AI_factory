// Compile-time CUDA launch profile shared by offline pricing and sampling.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#define AI_FACTORY_STRINGIFY_DETAIL(value) #value
#define AI_FACTORY_STRINGIFY(value) AI_FACTORY_STRINGIFY_DETAIL(value)

#ifndef AI_FACTORY_CUDA_TUNING_PROFILE_ID
#define AI_FACTORY_CUDA_TUNING_PROFILE_ID sm89_reference_v1
#endif
#ifndef AI_FACTORY_CUDA_MARKOVIAN_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_MARKOVIAN_THREADS_PER_BLOCK 512
#endif
#ifndef AI_FACTORY_CUDA_MARKOVIAN_COMPACT_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_MARKOVIAN_COMPACT_THREADS_PER_BLOCK 256
#endif
#ifndef AI_FACTORY_CUDA_N_FACTOR_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_N_FACTOR_THREADS_PER_BLOCK 256
#endif
#ifndef AI_FACTORY_CUDA_ANALYTICAL_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_ANALYTICAL_THREADS_PER_BLOCK 256
#endif
#ifndef AI_FACTORY_CUDA_EARLY_EXERCISE_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_EARLY_EXERCISE_THREADS_PER_BLOCK 128
#endif
#ifndef AI_FACTORY_CUDA_EARLY_EXERCISE_BLOCKS_PER_PRICE
#define AI_FACTORY_CUDA_EARLY_EXERCISE_BLOCKS_PER_PRICE 128
#endif
#ifndef AI_FACTORY_CUDA_FIXED_INCOME_LSM_BLOCKS_PER_PRICE
#define AI_FACTORY_CUDA_FIXED_INCOME_LSM_BLOCKS_PER_PRICE 64
#endif
#ifndef AI_FACTORY_CUDA_SAMPLE_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_SAMPLE_THREADS_PER_BLOCK 256
#endif
#ifndef AI_FACTORY_CUDA_SAMPLE_BLOCK_COUNT_LIMIT
#define AI_FACTORY_CUDA_SAMPLE_BLOCK_COUNT_LIMIT 4096
#endif
#ifndef AI_FACTORY_CUDA_VOLTERRA_PATH_CHUNK_SIZE
#define AI_FACTORY_CUDA_VOLTERRA_PATH_CHUNK_SIZE 65536
#endif
#ifndef AI_FACTORY_CUDA_VOLTERRA_PRICING_PATH_THREADS
#define AI_FACTORY_CUDA_VOLTERRA_PRICING_PATH_THREADS 256
#endif
#ifndef AI_FACTORY_CUDA_VOLTERRA_PRICING_FINALIZATION_THREADS
#define AI_FACTORY_CUDA_VOLTERRA_PRICING_FINALIZATION_THREADS 256
#endif

namespace ai_factory::workbench::offline::cuda_tuning {

inline constexpr std::string_view kProfileId =
    AI_FACTORY_STRINGIFY(AI_FACTORY_CUDA_TUNING_PROFILE_ID);
inline constexpr unsigned int kMarkovianThreadsPerBlock =
    AI_FACTORY_CUDA_MARKOVIAN_THREADS_PER_BLOCK;
inline constexpr unsigned int kMarkovianCompactThreadsPerBlock =
    AI_FACTORY_CUDA_MARKOVIAN_COMPACT_THREADS_PER_BLOCK;
inline constexpr unsigned int kNFactorThreadsPerBlock =
    AI_FACTORY_CUDA_N_FACTOR_THREADS_PER_BLOCK;
inline constexpr unsigned int kAnalyticalThreadsPerBlock =
    AI_FACTORY_CUDA_ANALYTICAL_THREADS_PER_BLOCK;
inline constexpr unsigned int kEarlyExerciseThreadsPerBlock =
    AI_FACTORY_CUDA_EARLY_EXERCISE_THREADS_PER_BLOCK;
inline constexpr std::size_t kEarlyExerciseBlocksPerPrice =
    AI_FACTORY_CUDA_EARLY_EXERCISE_BLOCKS_PER_PRICE;
inline constexpr std::size_t kFixedIncomeLsmBlocksPerPrice =
    AI_FACTORY_CUDA_FIXED_INCOME_LSM_BLOCKS_PER_PRICE;
inline constexpr unsigned int kSampleThreadsPerBlock =
    AI_FACTORY_CUDA_SAMPLE_THREADS_PER_BLOCK;
inline constexpr std::size_t kSampleBlockCountLimit =
    AI_FACTORY_CUDA_SAMPLE_BLOCK_COUNT_LIMIT;
inline constexpr std::size_t kVolterraPathChunkSize =
    AI_FACTORY_CUDA_VOLTERRA_PATH_CHUNK_SIZE;
inline constexpr unsigned int kVolterraPricingPathThreads =
    AI_FACTORY_CUDA_VOLTERRA_PRICING_PATH_THREADS;
inline constexpr unsigned int kVolterraPricingFinalizationThreads =
    AI_FACTORY_CUDA_VOLTERRA_PRICING_FINALIZATION_THREADS;

// Production pricing precision is independent of geometry and FFT chunk size.
// Smoke tests, warmups and reference experiments supply their own path counts.
inline constexpr std::size_t kProductionPathsPerPrice = 1U << 20U;
inline constexpr std::size_t kMonteCarloRowsPerLaunch = 4096U;
inline constexpr std::size_t kMonteCarloBlockCountLimit = 4096U;

enum class PricingFamily {
    closed_form,
    jamshidian,
    equity_exact_mc,
    equity_step_mc,
    fixed_income_mc,
    equity_lsm,
    gaussian_rate_lsm,
    terminal_forward_lsm,
    rough_n_factor,
    rough_fft,
};

inline constexpr std::string_view pricing_family_name(PricingFamily family) {
    switch (family) {
    case PricingFamily::closed_form: return "closed_form";
    case PricingFamily::jamshidian: return "jamshidian";
    case PricingFamily::equity_exact_mc: return "equity_exact_mc";
    case PricingFamily::equity_step_mc: return "equity_step_mc";
    case PricingFamily::fixed_income_mc: return "fixed_income_mc";
    case PricingFamily::equity_lsm: return "equity_lsm";
    case PricingFamily::gaussian_rate_lsm: return "gaussian_rate_lsm";
    case PricingFamily::terminal_forward_lsm: return "terminal_forward_lsm";
    case PricingFamily::rough_n_factor: return "rough_n_factor";
    case PricingFamily::rough_fft: return "rough_fft";
    }
    throw std::invalid_argument("Unknown pricing family.");
}

struct PricingIdentity {
    PricingFamily family = PricingFamily::equity_step_mc;
    std::string model;
    std::string product;
    std::string curve;
};

enum class PriceWorkDistribution { thread, block, lsm, fft };

struct PricingProfile {
    PriceWorkDistribution distribution = PriceWorkDistribution::block;
    unsigned int threads_per_block = kMarkovianThreadsPerBlock;
    std::size_t block_count_limit = kMonteCarloBlockCountLimit;
    std::size_t prices_per_launch = kMonteCarloRowsPerLaunch;
    std::size_t blocks_per_price = 0U;
    std::size_t path_chunk_size = 0U;
    std::string_view qualification = "reference_default_not_pair_qualified";
    std::string_view evidence = "docs/performance-reports/pricing-workload-scaling-sm89-2026-09-07.md";
};

// Only host-side choices live here. Existing kernels retain their arithmetic.
// Bounds are launch limits, not a claim that every inherited pair is optimal.
inline PricingProfile pricing_profile(PricingIdentity identity, std::size_t price_count = 0U) {
    PricingProfile profile;
    switch (identity.family) {
    case PricingFamily::closed_form:
        profile.distribution = PriceWorkDistribution::thread;
        profile.threads_per_block = kAnalyticalThreadsPerBlock;
        profile.block_count_limit = 0U;  // Dense grid.
        profile.prices_per_launch = 0U;  // Caller count, no artificial cap.
        profile.evidence = "docs/performance-reports/closed-form-price-count-scaling-sm89-2026-09-06.md";
        break;
    case PricingFamily::jamshidian:
        profile.distribution = identity.model == "cir"
            ? PriceWorkDistribution::block : PriceWorkDistribution::thread;
        profile.threads_per_block = identity.model == "cir"
            ? 128U : kAnalyticalThreadsPerBlock;
        profile.block_count_limit = 0U;
        profile.prices_per_launch = 0U;
        profile.evidence = "docs/performance-reports/jamshidian-strategy-scaling-sm89-2026-09-08.md";
        // The NUM-021 root correction is qualified numerically. Keep the
        // native geometry until timings of the corrected implementation are confirmed.
        if (identity.model == "cir" || identity.model == "vasicek") {
            profile.qualification = "native_geometry_pending_confirmation";
        } else if (kProfileId == "sm89_reference_v1" && price_count != 0U) {
            // The catalogue point was measured at exactly 1,000 prices.
            // Other sizes inherit the large-batch candidate, not an inferred
            // optimal crossover at 1,000 prices.
            const bool catalogue_batch = price_count == 1000U;
            profile.qualification = "candidate_pending_native_confirmation";
            if (identity.model == "cir_plus_plus") {
                profile.distribution = PriceWorkDistribution::block;
                profile.threads_per_block = catalogue_batch ? 256U : 64U;
                profile.block_count_limit = catalogue_batch ? 256U : 0U;
            } else if (identity.model == "hull_white"
                || (identity.model == "ornstein_uhlenbeck" && catalogue_batch)) {
                profile.distribution = PriceWorkDistribution::block;
                profile.threads_per_block = 128U;
            }
        }
        break;
    case PricingFamily::equity_exact_mc:
    case PricingFamily::equity_step_mc:
        if (identity.product == "cliquet")
            profile.threads_per_block = kMarkovianCompactThreadsPerBlock;
        break;
    case PricingFamily::fixed_income_mc:
        profile.threads_per_block = kMarkovianCompactThreadsPerBlock;
        profile.prices_per_launch = 256U;
        profile.evidence = "docs/performance-reports/g2-european-swaption-monte-carlo-sm89-2026-09-08.md";
        break;
    case PricingFamily::equity_lsm:
    case PricingFamily::gaussian_rate_lsm:
    case PricingFamily::terminal_forward_lsm:
        profile.distribution = PriceWorkDistribution::lsm;
        profile.threads_per_block = kEarlyExerciseThreadsPerBlock;
        profile.blocks_per_price = identity.family == PricingFamily::equity_lsm
            ? kEarlyExerciseBlocksPerPrice : kFixedIncomeLsmBlocksPerPrice;
        profile.block_count_limit = 0U;
        profile.prices_per_launch = 0U;  // Owned by the native VRAM planner.
        if (identity.family == PricingFamily::terminal_forward_lsm)
            profile.evidence = "docs/performance-reports/cir-forward-measure-comparison-sm89-2026-09-08.md";
        break;
    case PricingFamily::rough_n_factor:
        profile.threads_per_block = kNFactorThreadsPerBlock;
        break;
    case PricingFamily::rough_fft:
        profile.distribution = PriceWorkDistribution::fft;
        profile.threads_per_block = kVolterraPricingPathThreads;
        profile.block_count_limit = 0U;
        profile.prices_per_launch = 1U;
        profile.path_chunk_size = kVolterraPathChunkSize;
        break;
    }
    return profile;
}

inline nlohmann::ordered_json metadata(std::string_view family) {
    nlohmann::ordered_json result{
        {"profile_id", kProfileId},
        {"family", family},
        {"selection", "compile-time CMake profile"},
    };
    if (kProfileId == "sm89_reference_v1") {
        result["measured_on"] = "NVIDIA GeForce RTX 4090 Laptop GPU / SM89";
        result["portability"] =
            "safe reference values; benchmark before replacing on another GPU";
    } else {
        result["measured_on"] = "user-declared profile";
        result["portability"] =
            "valid only for the separately documented target environment";
    }
    if (family.find("volterra") != std::string_view::npos) {
        result["pricing_path_threads"] = kVolterraPricingPathThreads;
        result["pricing_finalization_threads"] =
            kVolterraPricingFinalizationThreads;
        result["pricing_path_chunk_size"] = kVolterraPathChunkSize;
    }
    return result;
}

}  // namespace ai_factory::workbench::offline::cuda_tuning

#undef AI_FACTORY_STRINGIFY
#undef AI_FACTORY_STRINGIFY_DETAIL
