// Compile-time CUDA launch profile shared by offline pricing and sampling.
#pragma once

#include <nlohmann/json.hpp>

#include <cstddef>
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
    return result;
}

}  // namespace ai_factory::workbench::offline::cuda_tuning

#undef AI_FACTORY_STRINGIFY
#undef AI_FACTORY_STRINGIFY_DETAIL
