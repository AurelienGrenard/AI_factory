// Replaceable compile-time tuning for terminal Monte Carlo sensitivities.
#pragma once

#include <cstddef>

#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_LAUNCH_BOUNDS
#define AI_FACTORY_CUDA_PRICE_GRADIENT_LAUNCH_BOUNDS 0
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_MONO_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_PRICE_GRADIENT_MONO_THREADS_PER_BLOCK 128
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_MONO_MIN_BLOCKS_PER_SM
#define AI_FACTORY_CUDA_PRICE_GRADIENT_MONO_MIN_BLOCKS_PER_SM 6
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_THREADS_PER_BLOCK
#define AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_THREADS_PER_BLOCK 128
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_MIN_BLOCKS_PER_SM
#define AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_MIN_BLOCKS_PER_SM 6
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_ROW_CHUNK_SIZE
#define AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_ROW_CHUNK_SIZE 16
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_PATH_CHUNK_SIZE
#define AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_PATH_CHUNK_SIZE 65536
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_WORKSPACE_MIB
#define AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_WORKSPACE_MIB 512
#endif
#ifndef AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_RESIDENT_WAVES
#define AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_RESIDENT_WAVES 4
#endif

namespace ai_factory::workbench::monte_carlo::price_gradients::tuning {

struct DefaultTerminalMonoTuning {
    static constexpr bool kLaunchBoundsEnabled =
        AI_FACTORY_CUDA_PRICE_GRADIENT_LAUNCH_BOUNDS != 0;
    static constexpr unsigned int kThreadsPerBlock =
        AI_FACTORY_CUDA_PRICE_GRADIENT_MONO_THREADS_PER_BLOCK;
    static constexpr unsigned int kMinimumBlocksPerMultiprocessor =
        AI_FACTORY_CUDA_PRICE_GRADIENT_MONO_MIN_BLOCKS_PER_SM;
};

struct DefaultTerminalNodeTuning {
    static constexpr bool kLaunchBoundsEnabled =
        AI_FACTORY_CUDA_PRICE_GRADIENT_LAUNCH_BOUNDS != 0;
    static constexpr unsigned int kThreadsPerBlock =
        AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_THREADS_PER_BLOCK;
    static constexpr unsigned int kMinimumBlocksPerMultiprocessor =
        AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_MIN_BLOCKS_PER_SM;
    static constexpr std::size_t kMaximumRowChunkSize =
        AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_ROW_CHUNK_SIZE;
    static constexpr std::size_t kPathChunkSize =
        AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_PATH_CHUNK_SIZE;
    static constexpr std::size_t kWorkspaceByteLimit =
        static_cast<std::size_t>(
            AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_WORKSPACE_MIB
        ) * 1024U * 1024U;
    static constexpr unsigned int kResidentWaves =
        AI_FACTORY_CUDA_PRICE_GRADIENT_NODE_RESIDENT_WAVES;
};

template<typename Profile>
inline constexpr bool valid_profile_v =
    Profile::kThreadsPerBlock >= 32U
    && Profile::kThreadsPerBlock <= 1024U
    && Profile::kThreadsPerBlock % 32U == 0U
    && Profile::kMinimumBlocksPerMultiprocessor >= 1U
    && Profile::kMinimumBlocksPerMultiprocessor <= 32U;

template<typename Profile>
inline constexpr bool valid_node_profile_v =
    valid_profile_v<Profile>
    && Profile::kMaximumRowChunkSize >= 1U
    && Profile::kPathChunkSize >= 32U
    && Profile::kWorkspaceByteLimit >= 1U
    && Profile::kResidentWaves >= 1U;

static_assert(valid_profile_v<DefaultTerminalMonoTuning>);
static_assert(valid_node_profile_v<DefaultTerminalNodeTuning>);

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::tuning
