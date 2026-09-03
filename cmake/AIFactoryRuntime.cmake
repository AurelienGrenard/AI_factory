# Runtime and offline dataset libraries.
add_library(ai_factory_cuda_tuning INTERFACE)
target_compile_definitions(ai_factory_cuda_tuning INTERFACE
    AI_FACTORY_CUDA_TUNING_PROFILE_ID=${AI_FACTORY_CUDA_TUNING_PROFILE_ID}
    AI_FACTORY_CUDA_MARKOVIAN_THREADS_PER_BLOCK=${AI_FACTORY_CUDA_MARKOVIAN_THREADS_PER_BLOCK}
    AI_FACTORY_CUDA_MARKOVIAN_COMPACT_THREADS_PER_BLOCK=${AI_FACTORY_CUDA_MARKOVIAN_COMPACT_THREADS_PER_BLOCK}
    AI_FACTORY_CUDA_N_FACTOR_THREADS_PER_BLOCK=${AI_FACTORY_CUDA_N_FACTOR_THREADS_PER_BLOCK}
    AI_FACTORY_CUDA_ANALYTICAL_THREADS_PER_BLOCK=${AI_FACTORY_CUDA_ANALYTICAL_THREADS_PER_BLOCK}
    AI_FACTORY_CUDA_EARLY_EXERCISE_THREADS_PER_BLOCK=${AI_FACTORY_CUDA_EARLY_EXERCISE_THREADS_PER_BLOCK}
    AI_FACTORY_CUDA_EARLY_EXERCISE_BLOCKS_PER_PRICE=${AI_FACTORY_CUDA_EARLY_EXERCISE_BLOCKS_PER_PRICE}
    AI_FACTORY_CUDA_FIXED_INCOME_LSM_BLOCKS_PER_PRICE=${AI_FACTORY_CUDA_FIXED_INCOME_LSM_BLOCKS_PER_PRICE}
    AI_FACTORY_CUDA_SAMPLE_THREADS_PER_BLOCK=${AI_FACTORY_CUDA_SAMPLE_THREADS_PER_BLOCK}
    AI_FACTORY_CUDA_SAMPLE_BLOCK_COUNT_LIMIT=${AI_FACTORY_CUDA_SAMPLE_BLOCK_COUNT_LIMIT}
    AI_FACTORY_CUDA_VOLTERRA_PATH_CHUNK_SIZE=${AI_FACTORY_CUDA_VOLTERRA_PATH_CHUNK_SIZE}
)

# Host-only JSON validation shared by runtime loaders and offline tools.
add_library(ai_factory_dataset_validation STATIC
    src/common/dataset_validation.cpp
)
target_include_directories(ai_factory_dataset_validation PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(
    ai_factory_dataset_validation PUBLIC nlohmann_json::nlohmann_json
)
target_compile_features(ai_factory_dataset_validation PUBLIC cxx_std_23)
target_compile_options(ai_factory_dataset_validation PRIVATE -O3)

# Independent offline stages: pure sampling, artifact I/O, parameter assembly,
# and price-result assembly.  Generator targets link only the stages they use.
function(ai_factory_add_offline_library target source)
    add_library(${target} STATIC ${source})
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${target} PUBLIC nlohmann_json::nlohmann_json)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_compile_options(${target} PRIVATE -O3)
endfunction()

ai_factory_add_offline_library(
    ai_factory_artifact_io tools/datasets/artifact_io.cpp
)
ai_factory_add_offline_library(
    ai_factory_sampling tools/datasets/sampling.cpp
)
ai_factory_add_offline_library(
    ai_factory_parameter_dataset tools/datasets/parameter_dataset.cpp
)
target_link_libraries(
    ai_factory_parameter_dataset PUBLIC ai_factory_artifact_io ai_factory_sampling
)
ai_factory_add_offline_library(
    ai_factory_price_dataset tools/datasets/price_dataset.cpp
)
target_link_libraries(
    ai_factory_price_dataset PUBLIC
    ai_factory_artifact_io
    ai_factory_dataset_validation
)
ai_factory_add_offline_library(
    ai_factory_sample_dataset tools/datasets/sample_dataset.cpp
)
target_link_libraries(
    ai_factory_sample_dataset PUBLIC ai_factory_artifact_io
)

# Compatibility aggregation is intentionally interface-only. New generators
# must depend on the precise stages selected below.
add_library(ai_factory_dataset_core INTERFACE)
target_link_libraries(ai_factory_dataset_core INTERFACE
    ai_factory_parameter_dataset
    ai_factory_price_dataset
    ai_factory_sample_dataset
)

# Parameter-only construction helpers stay outside price-generator builds.
set(_ai_factory_generation_helpers
    autocall_generation
    cliquet_generation
    g2_generation
    nelson_siegel_generation
    ornstein_uhlenbeck_generation
    range_accrual_generation
    svensson_generation
    vasicek_generation
)
set(_ai_factory_generation_targets)
foreach(helper IN LISTS _ai_factory_generation_helpers)
    set(target ai_factory_${helper})
    add_library(
        ${target} STATIC EXCLUDE_FROM_ALL tools/datasets/${helper}.cpp
    )
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${target} PUBLIC ai_factory_sampling)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_compile_options(${target} PRIVATE -O3)
    list(APPEND _ai_factory_generation_targets ${target})
endforeach()
