cmake_minimum_required(VERSION 3.20)

project(ai_factory_cuda_workbench LANGUAGES CXX CUDA)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.3)
    message(FATAL_ERROR "AI_factory requires CUDA Toolkit 13.3 or newer")
endif()

# CMake has recognized CUDA_STANDARD 23 since 3.20, but older releases do not
# provide NVIDIA's dialect flag. CUDA 13.3 accepts this flag directly.
if(NOT DEFINED CMAKE_CUDA23_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA23_STANDARD_COMPILE_OPTION "--std=c++23")
    set(CMAKE_CUDA23_EXTENSION_COMPILE_OPTION "--std=c++23")
endif()

include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json 3.2.0 REQUIRED)

option(
    AI_FACTORY_USE_CCACHE
    "Use ccache for repeated C++ and CUDA compilations when available"
    ON
)
if(AI_FACTORY_USE_CCACHE)
    find_program(AI_FACTORY_CCACHE_PROGRAM ccache)
    if(AI_FACTORY_CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${AI_FACTORY_CCACHE_PROGRAM}")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${AI_FACTORY_CCACHE_PROGRAM}")
        message(STATUS "Compiler cache: ${AI_FACTORY_CCACHE_PROGRAM}")
    else()
        message(STATUS "Compiler cache: ccache not found (optional)")
    endif()
endif()

# cuFFTDx is optional because mathDx is distributed separately from the CUDA
# toolkit. The fused rough-Bergomi pricer is enabled explicitly with the root
# of an extracted mathDx package.
set(
    AI_FACTORY_MATHDX_ROOT "" CACHE PATH
    "Root of NVIDIA mathDx (enables cuFFTDx Volterra pricers)"
)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Keep the architecture list overridable from the CMake command line. CUDA 13
# supports offline compilation for Turing (SM 7.5) and newer GPUs.
set(_cuda_workbench_default_architectures "75;86;89")
set(
    CUDA_WORKBENCH_ARCHITECTURES "${_cuda_workbench_default_architectures}"
    CACHE STRING
    "CUDA architectures (for example 89, 86, or 75;86;89)"
)
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_WORKBENCH_ARCHITECTURES}")

# Launch geometry is a replaceable build profile, not a numerical contract.
# The checked-in defaults are the measured SM89 reference values; deployments
# on another GPU should configure a new profile id and publish a separate
# performance baseline after resource and numerical checks.
set(AI_FACTORY_CUDA_TUNING_PROFILE_ID "sm89_reference_v1" CACHE STRING
    "Identifier recorded with CUDA launch tuning metadata")
set(AI_FACTORY_CUDA_MARKOVIAN_THREADS_PER_BLOCK 512 CACHE STRING
    "Threads per block for ordinary Markovian Monte Carlo")
set(AI_FACTORY_CUDA_MARKOVIAN_COMPACT_THREADS_PER_BLOCK 256 CACHE STRING
    "Threads per block for compact/ragged Markovian policies")
set(AI_FACTORY_CUDA_N_FACTOR_THREADS_PER_BLOCK 256 CACHE STRING
    "Threads per block for rough Markovian N-factor lifts")
set(AI_FACTORY_CUDA_ANALYTICAL_THREADS_PER_BLOCK 256 CACHE STRING
    "Threads per block for scalar analytical kernels")
set(AI_FACTORY_CUDA_EARLY_EXERCISE_THREADS_PER_BLOCK 128 CACHE STRING
    "Threads per block for Longstaff-Schwartz kernels")
set(AI_FACTORY_CUDA_EARLY_EXERCISE_BLOCKS_PER_PRICE 128 CACHE STRING
    "Blocks per equity early-exercise price")
set(AI_FACTORY_CUDA_FIXED_INCOME_LSM_BLOCKS_PER_PRICE 64 CACHE STRING
    "Blocks per fixed-income early-exercise price")
set(AI_FACTORY_CUDA_SAMPLE_THREADS_PER_BLOCK 256 CACHE STRING
    "Threads per block for Markovian model-sample kernels")
set(AI_FACTORY_CUDA_SAMPLE_BLOCK_COUNT_LIMIT 4096 CACHE STRING
    "Persistent block ceiling for model-sample kernels")
set(AI_FACTORY_CUDA_VOLTERRA_PATH_CHUNK_SIZE 65536 CACHE STRING
    "Reusable path chunk for Volterra FFT pricing")

if(NOT AI_FACTORY_CUDA_TUNING_PROFILE_ID MATCHES "^[A-Za-z0-9_.-]+$")
    message(FATAL_ERROR
        "AI_FACTORY_CUDA_TUNING_PROFILE_ID must contain only letters, digits, '.', '_' or '-'"
    )
endif()
foreach(variable IN ITEMS
    AI_FACTORY_CUDA_MARKOVIAN_THREADS_PER_BLOCK
    AI_FACTORY_CUDA_MARKOVIAN_COMPACT_THREADS_PER_BLOCK
    AI_FACTORY_CUDA_N_FACTOR_THREADS_PER_BLOCK
    AI_FACTORY_CUDA_ANALYTICAL_THREADS_PER_BLOCK
    AI_FACTORY_CUDA_EARLY_EXERCISE_THREADS_PER_BLOCK
    AI_FACTORY_CUDA_SAMPLE_THREADS_PER_BLOCK
)
    if(NOT ${variable} MATCHES "^[0-9]+$"
        OR ${variable} LESS 32 OR ${variable} GREATER 1024)
        message(FATAL_ERROR "${variable} must be an integer in [32, 1024]")
    endif()
    math(EXPR remainder "${${variable}} % 32")
    if(NOT remainder EQUAL 0)
        message(FATAL_ERROR "${variable} must be a multiple of one warp (32)")
    endif()
endforeach()
foreach(variable IN ITEMS
    AI_FACTORY_CUDA_EARLY_EXERCISE_BLOCKS_PER_PRICE
    AI_FACTORY_CUDA_FIXED_INCOME_LSM_BLOCKS_PER_PRICE
    AI_FACTORY_CUDA_SAMPLE_BLOCK_COUNT_LIMIT
    AI_FACTORY_CUDA_VOLTERRA_PATH_CHUNK_SIZE
)
    if(NOT ${variable} MATCHES "^[0-9]+$" OR ${variable} LESS 1)
        message(FATAL_ERROR "${variable} must be a positive integer")
    endif()
endforeach()

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

include(cmake/AIFactoryTargets.cmake)

# Register one host-only parameter generator with shared build settings.
add_custom_target(parameter_generators)
add_custom_target(price_generators)
add_custom_target(sample_generators)

function(ai_factory_collect_generation_dependencies output source)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_text)
    string(REGEX MATCHALL
        "tools/datasets/[a-z0-9_]+_generation\\.hpp"
        generation_headers
        "${source_text}"
    )
    set(dependencies)
    foreach(header IN LISTS generation_headers)
        string(REGEX REPLACE
            "tools/datasets/([^/]+)\\.hpp" "ai_factory_\\1"
            candidate "${header}"
        )
        if(TARGET ${candidate})
            list(APPEND dependencies ${candidate})
        endif()
    endforeach()
    string(REGEX MATCHALL
        "model/equity/(markovian|rough)/[a-z0-9_]+/dataset\\.hpp"
        equity_model_dataset_headers
        "${source_text}"
    )
    foreach(header IN LISTS equity_model_dataset_headers)
        string(REGEX REPLACE
            "model/equity/(markovian|rough)/([^/]+)/dataset\\.hpp"
            "ai_factory_equity_\\2_dataset"
            candidate
            "${header}"
        )
        if(TARGET ${candidate})
            list(APPEND dependencies ${candidate})
        endif()
    endforeach()
    string(REGEX MATCHALL
        "model/fixed_income/[a-z0-9_]+/dataset\\.hpp"
        fixed_income_model_dataset_headers
        "${source_text}"
    )
    foreach(header IN LISTS fixed_income_model_dataset_headers)
        string(REGEX REPLACE
            "model/fixed_income/([^/]+)/dataset\\.hpp"
            "ai_factory_fixed_income_\\1_dataset"
            candidate
            "${header}"
        )
        if(TARGET ${candidate})
            list(APPEND dependencies ${candidate})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES dependencies)
    set(${output} ${dependencies} PARENT_SCOPE)
endfunction()

function(add_parameter_generator target source)
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    ai_factory_collect_generation_dependencies(dependencies ${source})
    target_link_libraries(
        ${target} PRIVATE
        ai_factory_parameter_dataset
        ai_factory_dataset_validation
        ${dependencies}
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    add_dependencies(parameter_generators ${target})
endfunction()

# Register one CUDA price generator with shared build settings.
function(add_price_generator target source)
    if(source MATCHES
        "catalog/model/equity/(rough_heston|quadratic_rough_heston)/prices/")
        # Host preparation includes the CUDA dynamics contract that defines
        # PreparedDynamics<N>; compile these generated recipes with nvcc even
        # though their entry-point extension remains generator.cpp.
        set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    ai_factory_collect_source_dependencies(dependencies ${source})
    if(NOT dependencies)
        message(FATAL_ERROR
            "No compiled pricing unit found for generator ${target}: ${source}"
        )
    endif()
    target_link_libraries(
        ${target} PRIVATE ${dependencies} ai_factory_price_dataset
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED YES
    )
    add_dependencies(price_generators ${target})
endfunction()

# Register one model-sample generator against its thin CUDA model binding.
function(add_sample_generator target source)
    if(source MATCHES
        "catalog/model/equity/(rough_heston|quadratic_rough_heston)/samples/")
        # Host N-factor preparation includes CUDA-decorated dynamics types.
        set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    ai_factory_collect_source_dependencies(dependencies ${source})
    if(NOT dependencies)
        message(FATAL_ERROR
            "No compiled sample unit found for generator ${target}: ${source}"
        )
    endif()
    target_link_libraries(
        ${target} PRIVATE ${dependencies} ai_factory_sample_dataset
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED YES
    )
    add_dependencies(sample_generators ${target})
endfunction()

# Derive a stable executable name from the versioned catalog recipe id. For
# example `heston_01__european_calls_01__01` remains the public target
# `generate_heston_european_calls_01`; fitted recipes include their curve id.
function(ai_factory_catalog_generator_target output source)
    get_filename_component(recipe_directory "${source}" DIRECTORY)
    get_filename_component(recipe_id "${recipe_directory}" NAME)
    if(source MATCHES "/prices/")
        string(REPLACE "__" ";" components "${recipe_id}")
        list(POP_BACK components version)
        set(target generate)
        foreach(component IN LISTS components)
            string(REGEX REPLACE "_[0-9]+$" "" component "${component}")
            string(APPEND target "_${component}")
        endforeach()
        string(APPEND target "_${version}")
    elseif(source MATCHES
        "catalog/model/(equity|fixed_income)/([^/]+)/samples/")
        set(target "generate_${CMAKE_MATCH_2}_${recipe_id}")
    else()
        set(target "generate_${recipe_id}")
    endif()
    set(${output} "${target}" PARENT_SCOPE)
endfunction()

# The generated manifest owns the equity recipes and their implementation
# matrix. Tree discovery registers fixed-income and parameter recipes while
# the capability checker verifies their declared matrix.
file(GLOB_RECURSE _ai_factory_catalog_generators
    CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/catalog/*/generator.cpp"
)
list(SORT _ai_factory_catalog_generators)
foreach(source IN LISTS _ai_factory_catalog_generators)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_text)
    if(NOT AI_FACTORY_MATHDX_ROOT
        AND source_text MATCHES
            "model/equity/rough/(rough_bergomi|rough_sabr|log_modulated_rough_bergomi|rough_stein_stein)/")
        continue()
    endif()
    ai_factory_catalog_generator_target(target "${source}")
    if(TARGET ${target})
        message(FATAL_ERROR
            "Duplicate catalog generator target ${target}: ${source}"
        )
    endif()
    if(source MATCHES "/prices/")
        add_price_generator(${target} "${source}")
    elseif(source MATCHES "/samples/")
        add_sample_generator(${target} "${source}")
    else()
        add_parameter_generator(${target} "${source}")
    endif()
endforeach()

add_custom_target(performance_benchmarks)

add_executable(
    ai_factory_generic_kernel_benchmark EXCLUDE_FROM_ALL
    validation/performance/generic_kernel_benchmark.cu
)
ai_factory_configure_cuda_library(ai_factory_generic_kernel_benchmark)
target_link_libraries(
    ai_factory_generic_kernel_benchmark PRIVATE
    ai_factory_runtime
    ai_factory_equity_black_scholes_european_option
    ai_factory_equity_bates_phoenix_autocall
    ai_factory_equity_bates_phoenix_memory_autocall
    ai_factory_fixed_income_hull_white_nelson_siegel_european_swaption
)
add_dependencies(
    performance_benchmarks ai_factory_generic_kernel_benchmark
)

add_executable(
    ai_factory_early_exercise_benchmark EXCLUDE_FROM_ALL
    validation/performance/early_exercise_benchmark.cu
)
ai_factory_configure_cuda_library(ai_factory_early_exercise_benchmark)
target_link_libraries(
    ai_factory_early_exercise_benchmark PRIVATE
    ai_factory_runtime
    ai_factory_equity_black_scholes_american_option
    ai_factory_equity_heston_american_option
    ai_factory_fixed_income_ornstein_uhlenbeck_bermudan_swaption
    ai_factory_fixed_income_g2_bermudan_swaption
)
add_dependencies(
    performance_benchmarks ai_factory_early_exercise_benchmark
)

add_library(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment
    STATIC EXCLUDE_FROM_ALL
    src/model/fixed_income/cir/product/european_swaption.cu
)
ai_factory_configure_cuda_library(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment
)
target_compile_definitions(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment PRIVATE
    AI_FACTORY_NCX2_FORCE_INLINE=1
)
target_link_libraries(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment PUBLIC
    ai_factory_runtime
    ai_factory_fixed_income_cir_dataset
    ai_factory_product_european_swaption_dataset
)

foreach(variant IN ITEMS inline noinline)
    set(target ai_factory_cir_kernel_benchmark_${variant})
    add_executable(
        ${target} EXCLUDE_FROM_ALL
        validation/performance/cir_kernel_benchmark.cu
    )
    ai_factory_configure_cuda_library(${target})
    target_compile_definitions(
        ${target} PRIVATE
        AI_FACTORY_CIR_BENCHMARK_VARIANT="${variant}"
    )
    if(variant STREQUAL "inline")
        target_link_libraries(
            ${target} PRIVATE
            ai_factory_fixed_income_cir_european_swaption_inline_experiment
        )
    else()
        target_link_libraries(
            ${target} PRIVATE
            ai_factory_fixed_income_cir_european_swaption
        )
    endif()
    add_dependencies(performance_benchmarks ${target})
endforeach()

if(AI_FACTORY_MATHDX_ROOT)
    add_executable(
        ai_factory_model_sample_benchmark EXCLUDE_FROM_ALL
        validation/performance/model_sample_benchmark.cu
    )
    ai_factory_configure_cuda_library(ai_factory_model_sample_benchmark)
    target_link_libraries(
        ai_factory_model_sample_benchmark PRIVATE
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_black_scholes_sample
        ai_factory_equity_heston_sample
        ai_factory_equity_rough_heston_sample
        ai_factory_equity_rough_bergomi_sample
    )
    add_dependencies(
        performance_benchmarks ai_factory_model_sample_benchmark
    )

    add_library(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        STATIC EXCLUDE_FROM_ALL
        src/model/equity/rough/rough_bergomi/product/european_option.cu
    )
    ai_factory_configure_cuda_library(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
    )
    target_compile_definitions(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        PRIVATE AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT=32
    )
    target_link_libraries(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        PUBLIC
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_rough_bergomi_dataset
        ai_factory_product_european_option_dataset
    )

    add_executable(
        ai_factory_volterra_kernel_benchmark EXCLUDE_FROM_ALL
        validation/volterra/kernel_benchmark.cu
    )
    ai_factory_configure_cuda_library(ai_factory_volterra_kernel_benchmark)
    target_link_libraries(
        ai_factory_volterra_kernel_benchmark PRIVATE
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_heston_european_option
        ai_factory_equity_rough_heston_european_option
        ai_factory_equity_rough_bergomi_european_option
        ai_factory_equity_rough_sabr_european_option
    )
    add_dependencies(
        performance_benchmarks ai_factory_volterra_kernel_benchmark
    )

    add_executable(
        ai_factory_volterra_direct_kernel_benchmark EXCLUDE_FROM_ALL
        validation/volterra/kernel_benchmark.cu
    )
    ai_factory_configure_cuda_library(
        ai_factory_volterra_direct_kernel_benchmark
    )
    target_compile_definitions(
        ai_factory_volterra_direct_kernel_benchmark PRIVATE
        AI_FACTORY_VOLTERRA_BENCHMARK_VARIANT="direct_experiment"
    )
    target_link_libraries(
        ai_factory_volterra_direct_kernel_benchmark PRIVATE
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_heston_european_option
        ai_factory_equity_rough_heston_european_option
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        ai_factory_equity_rough_sabr_european_option
    )
    add_dependencies(
        performance_benchmarks ai_factory_volterra_direct_kernel_benchmark
    )
endif()

if(BUILD_TESTING)
    add_custom_target(ai_factory_tests)
    add_custom_target(ai_factory_host_tests)
    add_custom_target(ai_factory_cuda_tests)
    add_custom_target(common_cuda_tests)
    add_custom_target(equity_tests)
    add_custom_target(fixed_income_tests)
    add_dependencies(ai_factory_tests
        ai_factory_host_tests
        ai_factory_cuda_tests
    )
    add_dependencies(ai_factory_cuda_tests
        common_cuda_tests
        equity_tests
        fixed_income_tests
    )

    option(
        AI_FACTORY_QUANTLIB_VALIDATION
        "Run independent price-dataset checks through QuantLib Python"
        ON
    )
    option(
        AI_FACTORY_QUANTLIB_EXOTIC_VALIDATION
        "Run slow QuantLib validation for Heston/Bates path and American options"
        OFF
    )

    add_executable(
        test_dataset_catalog
        EXCLUDE_FROM_ALL
        tests/catalog_contract_test.cpp
    )
    target_include_directories(test_dataset_catalog PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(
        test_dataset_catalog
        PRIVATE ai_factory_dataset_core
    )
    target_compile_features(test_dataset_catalog PRIVATE cxx_std_23)
    add_dependencies(ai_factory_host_tests test_dataset_catalog)
    add_test(
        NAME dataset_catalog
        COMMAND test_dataset_catalog
    )
    set_tests_properties(dataset_catalog PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;catalog"
    )

    function(add_offline_stage_test stage library)
        set(target test_${stage}_stage)
        add_executable(
            ${target} EXCLUDE_FROM_ALL tests/${stage}_stage_test.cpp
        )
        target_link_libraries(${target} PRIVATE ${library})
        target_compile_features(${target} PRIVATE cxx_std_23)
        add_dependencies(ai_factory_host_tests ${target})
        add_test(NAME ${stage}_stage COMMAND ${target})
        set_tests_properties(${stage}_stage PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;offline;${stage}"
            TIMEOUT 30
        )
    endfunction()
    add_offline_stage_test(sampling ai_factory_sampling)
    add_offline_stage_test(artifact_io ai_factory_artifact_io)
    add_offline_stage_test(parameter_dataset ai_factory_parameter_dataset)
    add_offline_stage_test(sample_dataset ai_factory_sample_dataset)
    add_offline_stage_test(price_dataset ai_factory_price_dataset)

    add_executable(
        test_dataset_loaders EXCLUDE_FROM_ALL tests/dataset_loaders_test.cpp
    )
    ai_factory_collect_source_dependencies(
        dataset_loader_dependencies tests/dataset_loaders_test.cpp
    )
    target_link_libraries(
        test_dataset_loaders PRIVATE ${dataset_loader_dependencies}
    )
    target_compile_features(test_dataset_loaders PRIVATE cxx_std_23)
    add_dependencies(ai_factory_host_tests test_dataset_loaders)
    add_test(NAME dataset_loaders COMMAND test_dataset_loaders)
    set_tests_properties(dataset_loaders PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;dataset"
    )

    add_executable(
        test_rough_sabr_dataset_loader
        EXCLUDE_FROM_ALL
        tests/rough_sabr_dataset_loader_test.cpp
    )
    target_link_libraries(
        test_rough_sabr_dataset_loader
        PRIVATE ai_factory_equity_rough_sabr_dataset
    )
    target_compile_features(test_rough_sabr_dataset_loader PRIVATE cxx_std_23)
    add_dependencies(ai_factory_host_tests test_rough_sabr_dataset_loader)
    add_test(
        NAME rough_sabr_dataset_loader
        COMMAND test_rough_sabr_dataset_loader
    )
    set_tests_properties(rough_sabr_dataset_loader PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;dataset;rough_sabr"
    )

    # Register independent QuantLib references only when its binding is present.
    if(AI_FACTORY_QUANTLIB_VALIDATION)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_test(
                NAME quantlib_validation_metrics
                COMMAND
                    ${Python3_EXECUTABLE} -m unittest
                    validation.quantlib.tests.test_price_validation
            )
            set_tests_properties(
                quantlib_validation_metrics
                PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    LABELS "workbench;validation;quantlib"
            )
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import QuantLib"
                RESULT_VARIABLE _quantlib_python_status
                OUTPUT_QUIET
                ERROR_QUIET
            )
            if(_quantlib_python_status EQUAL 0)
                add_test(
                    NAME quantlib_bermudan_swaption_adapter
                    COMMAND
                        ${Python3_EXECUTABLE} -m unittest
                        validation.quantlib.tests.test_bermudan_swaption
                )
                set_tests_properties(
                    quantlib_bermudan_swaption_adapter
                    PROPERTIES
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        LABELS "workbench;validation;quantlib;bermudan_swaption"
                )
                function(add_quantlib_validation_test name module dataset label)
                    add_test(
                        NAME quantlib_${name}
                        COMMAND
                            ${Python3_EXECUTABLE} -m ${module} ${dataset} ${ARGN}
                    )
                    set_tests_properties(
                        quantlib_${name}
                        PROPERTIES
                            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                            LABELS "workbench;validation;quantlib;${label}"
                    )
                endfunction()

                foreach(product IN ITEMS
                    zero_coupon_bond_call
                    zero_coupon_bond_put
                    caplet
                    floorlet
                )
                    if(product STREQUAL "zero_coupon_bond_call")
                        set(product_folder zero_coupon_bond_calls)
                    elseif(product STREQUAL "zero_coupon_bond_put")
                        set(product_folder zero_coupon_bond_puts)
                    elseif(product STREQUAL "caplet")
                        set(product_folder caplets)
                    else()
                        set(product_folder floorlets)
                    endif()

                    add_quantlib_validation_test(
                        ornstein_uhlenbeck_${product}
                        validation.quantlib.model.fixed_income.ornstein_uhlenbeck.${product}
                        datasets/model/fixed_income/ornstein_uhlenbeck/prices/${product_folder}/ornstein_uhlenbeck_01__${product_folder}_01__01.json
                        ornstein_uhlenbeck
                    )
                    add_quantlib_validation_test(
                        vasicek_${product}
                        validation.quantlib.model.fixed_income.vasicek.${product}
                        datasets/model/fixed_income/vasicek/prices/${product_folder}/vasicek_01__${product_folder}_01__01.json
                        vasicek
                    )
                    add_quantlib_validation_test(
                        cir_${product}
                        validation.quantlib.model.fixed_income.cir.${product}
                        datasets/model/fixed_income/cir/prices/${product_folder}/cir_01__${product_folder}_01__01.json
                        cir
                    )
                    add_quantlib_validation_test(
                        hull_white_nelson_siegel_${product}
                        validation.quantlib.model.fixed_income.hull_white.nelson_siegel.${product}
                        datasets/model/fixed_income/hull_white/prices/nelson_siegel/${product_folder}/hull_white_01__nelson_siegel_01__${product_folder}_01__01.json
                        hull_white
                    )
                    add_quantlib_validation_test(
                        hull_white_svensson_${product}
                        validation.quantlib.model.fixed_income.hull_white.svensson.${product}
                        datasets/model/fixed_income/hull_white/prices/svensson/${product_folder}/hull_white_01__svensson_01__${product_folder}_01__01.json
                        hull_white
                    )
                    add_quantlib_validation_test(
                        g2_${product}
                        validation.quantlib.model.fixed_income.g2.${product}
                        datasets/model/fixed_income/g2/prices/${product_folder}/g2_01__${product_folder}_01__01.json
                        g2
                    )
                    add_quantlib_validation_test(
                        g2_plus_plus_nelson_siegel_${product}
                        validation.quantlib.model.fixed_income.g2_plus_plus.nelson_siegel.${product}
                        datasets/model/fixed_income/g2_plus_plus/prices/nelson_siegel/${product_folder}/g2_plus_plus_01__nelson_siegel_01__${product_folder}_01__01.json
                        g2_plus_plus
                    )
                    add_quantlib_validation_test(
                        g2_plus_plus_svensson_${product}
                        validation.quantlib.model.fixed_income.g2_plus_plus.svensson.${product}
                        datasets/model/fixed_income/g2_plus_plus/prices/svensson/${product_folder}/g2_plus_plus_01__svensson_01__${product_folder}_01__01.json
                        g2_plus_plus
                    )
                endforeach()

                add_quantlib_validation_test(
                    heston_european_call
                    validation.quantlib.model.equity.heston.european_call
                    datasets/model/equity/heston/prices/european_calls/heston_01__european_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_european_put
                    validation.quantlib.model.equity.heston.european_put
                    datasets/model/equity/heston/prices/european_puts/heston_01__european_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_straddle
                    validation.quantlib.model.equity.heston.straddle
                    datasets/model/equity/heston/prices/straddles/heston_01__straddles_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_digital_call
                    validation.quantlib.model.equity.heston.digital_call
                    datasets/model/equity/heston/prices/digital_calls/heston_01__digital_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_digital_put
                    validation.quantlib.model.equity.heston.digital_put
                    datasets/model/equity/heston/prices/digital_puts/heston_01__digital_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_asset_or_nothing_call
                    validation.quantlib.model.equity.heston.asset_or_nothing_call
                    datasets/model/equity/heston/prices/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_asset_or_nothing_put
                    validation.quantlib.model.equity.heston.asset_or_nothing_put
                    datasets/model/equity/heston/prices/asset_or_nothing_puts/heston_01__asset_or_nothing_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_gap_call
                    validation.quantlib.model.equity.heston.gap_call
                    datasets/model/equity/heston/prices/gap_calls/heston_01__gap_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_gap_put
                    validation.quantlib.model.equity.heston.gap_put
                    datasets/model/equity/heston/prices/gap_puts/heston_01__gap_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    cev_european_call
                    validation.quantlib.model.equity.cev.european_call
                    datasets/model/equity/cev/prices/european_calls/cev_01__european_calls_01__01.json
                    cev
                )
                add_quantlib_validation_test(
                    cev_european_put
                    validation.quantlib.model.equity.cev.european_put
                    datasets/model/equity/cev/prices/european_puts/cev_01__european_puts_01__01.json
                    cev
                )

                function(add_bates_validation_test product folder)
                    add_quantlib_validation_test(
                        bates_${product}
                        validation.quantlib.model.equity.bates.${product}
                        datasets/model/equity/bates/prices/${folder}/bates_01__${folder}_01__01.json
                        bates
                    )
                endfunction()

                add_bates_validation_test(european_call european_calls)
                add_bates_validation_test(european_put european_puts)
                add_bates_validation_test(straddle straddles)
                add_bates_validation_test(digital_call digital_calls)
                add_bates_validation_test(digital_put digital_puts)
                add_bates_validation_test(
                    asset_or_nothing_call asset_or_nothing_calls
                )
                add_bates_validation_test(
                    asset_or_nothing_put asset_or_nothing_puts
                )
                add_bates_validation_test(gap_call gap_calls)
                add_bates_validation_test(gap_put gap_puts)

                execute_process(
                    COMMAND ${Python3_EXECUTABLE} -c "import scipy"
                    RESULT_VARIABLE _scipy_python_status
                    OUTPUT_QUIET
                    ERROR_QUIET
                )
                if(_scipy_python_status EQUAL 0)
                    function(add_variance_gamma_validation_test product folder)
                        set(_vg_dataset
                            datasets/model/equity/variance_gamma/prices/${folder}/variance_gamma_01__${folder}_01__01.json
                        )
                        if(EXISTS "${CMAKE_SOURCE_DIR}/${_vg_dataset}")
                            add_quantlib_validation_test(
                                variance_gamma_${product}
                                validation.quantlib.model.equity.variance_gamma.${product}
                                ${_vg_dataset}
                                variance_gamma
                            )
                        endif()
                    endfunction()
                    add_variance_gamma_validation_test(
                        european_call european_calls
                    )
                    add_variance_gamma_validation_test(
                        european_put european_puts
                    )
                    add_variance_gamma_validation_test(straddle straddles)
                    add_variance_gamma_validation_test(digital_call digital_calls)
                    add_variance_gamma_validation_test(digital_put digital_puts)
                    add_variance_gamma_validation_test(
                        asset_or_nothing_call asset_or_nothing_calls
                    )
                    add_variance_gamma_validation_test(
                        asset_or_nothing_put asset_or_nothing_puts
                    )
                    add_variance_gamma_validation_test(gap_call gap_calls)
                    add_variance_gamma_validation_test(gap_put gap_puts)
                else()
                    message(STATUS
                        "SciPy is unavailable: Variance-Gamma reference tests disabled"
                    )
                endif()
                if(AI_FACTORY_QUANTLIB_EXOTIC_VALIDATION)
                    add_quantlib_validation_test(
                        heston_geometric_asian_call
                        validation.quantlib.model.equity.heston.geometric_asian_call
                        datasets/model/equity/heston/prices/geometric_asian_calls/heston_01__geometric_asian_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_geometric_asian_put
                        validation.quantlib.model.equity.heston.geometric_asian_put
                        datasets/model/equity/heston/prices/geometric_asian_puts/heston_01__geometric_asian_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_asian_call
                        validation.quantlib.model.equity.heston.asian_call
                        datasets/model/equity/heston/prices/asian_calls/heston_01__asian_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_asian_put
                        validation.quantlib.model.equity.heston.asian_put
                        datasets/model/equity/heston/prices/asian_puts/heston_01__asian_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_and_out_call
                        validation.quantlib.model.equity.heston.up_and_out_call
                        datasets/model/equity/heston/prices/up_and_out_calls/heston_01__up_and_out_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_down_and_out_put
                        validation.quantlib.model.equity.heston.down_and_out_put
                        datasets/model/equity/heston/prices/down_and_out_puts/heston_01__down_and_out_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_and_in_call
                        validation.quantlib.model.equity.heston.up_and_in_call
                        datasets/model/equity/heston/prices/up_and_in_calls/heston_01__up_and_in_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_down_and_in_put
                        validation.quantlib.model.equity.heston.down_and_in_put
                        datasets/model/equity/heston/prices/down_and_in_puts/heston_01__down_and_in_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_one_touch
                        validation.quantlib.model.equity.heston.up_one_touch
                        datasets/model/equity/heston/prices/up_one_touches/heston_01__up_one_touches_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_no_touch
                        validation.quantlib.model.equity.heston.up_no_touch
                        datasets/model/equity/heston/prices/up_no_touches/heston_01__up_no_touches_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_double_knock_out_call
                        validation.quantlib.model.equity.heston.double_knock_out_call
                        datasets/model/equity/heston/prices/double_knock_out_calls/heston_01__double_knock_out_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_double_knock_out_put
                        validation.quantlib.model.equity.heston.double_knock_out_put
                        datasets/model/equity/heston/prices/double_knock_out_puts/heston_01__double_knock_out_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_athena_autocall
                        validation.quantlib.model.equity.heston.athena_autocall
                        datasets/model/equity/heston/prices/athena_autocalls/heston_01__athena_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_phoenix_autocall
                        validation.quantlib.model.equity.heston.phoenix_autocall
                        datasets/model/equity/heston/prices/phoenix_autocalls/heston_01__phoenix_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_phoenix_memory_autocall
                        validation.quantlib.model.equity.heston.phoenix_memory_autocall
                        datasets/model/equity/heston/prices/phoenix_memory_autocalls/heston_01__phoenix_memory_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_cliquet
                        validation.quantlib.model.equity.heston.cliquet
                        datasets/model/equity/heston/prices/cliquets/heston_01__cliquets_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_range_accrual
                        validation.quantlib.model.equity.heston.range_accrual
                        datasets/model/equity/heston/prices/range_accruals/heston_01__range_accruals_01__01.json
                        heston
                    )
                    add_bates_validation_test(
                        geometric_asian_call geometric_asian_calls
                    )
                    add_bates_validation_test(
                        geometric_asian_put geometric_asian_puts
                    )
                    add_bates_validation_test(asian_call asian_calls)
                    add_bates_validation_test(asian_put asian_puts)
                    add_bates_validation_test(
                        forward_start_call forward_start_calls
                    )
                    add_bates_validation_test(
                        forward_start_put forward_start_puts
                    )
                    add_bates_validation_test(
                        up_and_out_call up_and_out_calls
                    )
                    add_bates_validation_test(
                        down_and_out_put down_and_out_puts
                    )
                    add_bates_validation_test(up_and_in_call up_and_in_calls)
                    add_bates_validation_test(down_and_in_put down_and_in_puts)
                    add_bates_validation_test(up_one_touch up_one_touches)
                    add_bates_validation_test(up_no_touch up_no_touches)
                    add_bates_validation_test(
                        double_knock_out_call double_knock_out_calls
                    )
                    add_bates_validation_test(
                        double_knock_out_put double_knock_out_puts
                    )
                    add_bates_validation_test(athena_autocall athena_autocalls)
                    add_bates_validation_test(
                        phoenix_autocall phoenix_autocalls
                    )
                    add_bates_validation_test(
                        phoenix_memory_autocall phoenix_memory_autocalls
                    )
                    add_bates_validation_test(cliquet cliquets)
                    add_bates_validation_test(range_accrual range_accruals)
                endif()
            else()
                message(STATUS
                    "QuantLib Python is unavailable: reference tests disabled"
                )
            endif()
        endif()
    endif()

    # Premia is the preferred specialized reference when its bundled runtime,
    # Wine, and the MinGW cross-compiler are all available. Every registered
    # test uses the same one-module-per-payoff CLI as the QuantLib validators.
    option(
        AI_FACTORY_PREMIA_VALIDATION
        "Run independent price-dataset checks through bundled Premia"
        ON
    )
    if(AI_FACTORY_PREMIA_VALIDATION)
        find_package(Python3 COMPONENTS Interpreter QUIET)
    endif()
    if(AI_FACTORY_PREMIA_VALIDATION AND Python3_Interpreter_FOUND)
        find_program(AI_FACTORY_WINE_EXECUTABLE wine)
        find_program(
            AI_FACTORY_MINGW_EXECUTABLE
            x86_64-w64-mingw32-gcc-posix
        )
        if(
            AI_FACTORY_WINE_EXECUTABLE
            AND AI_FACTORY_MINGW_EXECUTABLE
            AND EXISTS
                "${CMAKE_SOURCE_DIR}/validation/premia/premia-19-win64"
        )
            function(add_equity_model_unified_validation model product)
                if(product STREQUAL "up_no_touch")
                    set(folder up_no_touches)
                elseif(product STREQUAL "up_one_touch")
                    set(folder up_one_touches)
                else()
                    set(folder ${product}s)
                endif()
                add_test(
                    NAME validation_${model}_${product}
                    COMMAND
                        ${Python3_EXECUTABLE} -m
                        validation.model.equity.${model}.${product}
                        ${CMAKE_SOURCE_DIR}/datasets/model/equity/${model}/prices/${folder}/${model}_01__${folder}_01__01.json
                        ${CMAKE_CURRENT_BINARY_DIR}/validation-reports/${model}_${product}.json
                )
                set_tests_properties(
                    validation_${model}_${product}
                    PROPERTIES
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        LABELS "workbench;validation;premia;quantlib;${model}"
                        TIMEOUT 1800
                )
            endfunction()
            foreach(model IN ITEMS merton kou)
                foreach(product IN ITEMS
                    asian_call asian_put
                    asset_or_nothing_call asset_or_nothing_put
                    athena_autocall cliquet digital_call digital_put
                    double_knock_out_call double_knock_out_put
                    down_and_in_put down_and_out_put
                    european_call european_put
                    forward_start_call forward_start_put
                    gap_call gap_put
                    geometric_asian_call geometric_asian_put
                    lookback_option phoenix_autocall phoenix_memory_autocall
                    range_accrual straddle up_and_in_call up_and_out_call
                    up_no_touch up_one_touch
                )
                    add_equity_model_unified_validation(${model} ${product})
                endforeach()
            endforeach()
            foreach(model IN ITEMS heston bates)
                foreach(product IN ITEMS
                    american_call american_put asian_call asian_put
                    asset_or_nothing_call asset_or_nothing_put
                    athena_autocall cliquet digital_call digital_put
                    double_knock_out_call double_knock_out_put
                    down_and_in_put down_and_out_put
                    european_call european_put
                    forward_start_call forward_start_put
                    gap_call gap_put
                    geometric_asian_call geometric_asian_put
                    lookback_option phoenix_autocall phoenix_memory_autocall
                    range_accrual straddle up_and_in_call up_and_out_call
                    up_no_touch up_one_touch
                )
                    add_equity_model_unified_validation(${model} ${product})
                endforeach()
            endforeach()

        else()
            message(STATUS
                "Premia runtime prerequisites unavailable: reference tests disabled"
            )
        endif()
    endif()

    # Persistent publication checks consume immutable references and never
    # rerun Premia, QuantLib, or Wine. Keep them available on any Python host.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        if(AI_FACTORY_MATHDX_ROOT)
            add_custom_target(
                performance_regression_gate
                COMMAND
                    ${Python3_EXECUTABLE}
                    ${CMAKE_SOURCE_DIR}/validation/performance/run_baseline.py
                    --baseline
                    ${CMAKE_SOURCE_DIR}/validation/performance/baseline_sm89_v1.json
                    --build-dir ${CMAKE_BINARY_DIR}
                    --output
                    ${CMAKE_BINARY_DIR}/performance_candidate_sm89_v1.ndjson
                DEPENDS performance_benchmarks
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                USES_TERMINAL
            )
        endif()
        add_test(
            NAME performance_baseline_checker
            COMMAND
                ${Python3_EXECUTABLE} -m unittest
                validation.performance.test_check_baseline
        )
        set_tests_properties(performance_baseline_checker PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;performance;baseline"
            TIMEOUT 30
        )
        add_test(
            NAME pricing_binding_codegen
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/codegen/pricing_bindings/generate.py
                --family all
                --output
                ${CMAKE_BINARY_DIR}/generated/pricing_bindings
                --compare-root ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties(pricing_binding_codegen PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;codegen;equity"
            TIMEOUT 60
        )
        add_test(
            NAME pricing_capability_manifest
            COMMAND
                ${Python3_EXECUTABLE} -m unittest
                tools.codegen.pricing_bindings.test_capability_manifest
        )
        set_tests_properties(pricing_capability_manifest PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;codegen;architecture"
            TIMEOUT 30
        )
        add_test(
            NAME model_source_layout
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/cuda/check_model_layout.py
        )
        set_tests_properties(model_source_layout PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;codegen;architecture"
            TIMEOUT 30
        )
        add_test(
            NAME catalog_generator_boundaries
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/cuda/check_catalog_generators.py
        )
        set_tests_properties(catalog_generator_boundaries PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;catalog;architecture"
            TIMEOUT 30
        )

        function(add_cached_black_scholes_validation product folder)
            add_test(
                NAME validation_black_scholes_${product}
                COMMAND
                    ${Python3_EXECUTABLE} -m
                    validation.model.equity.black_scholes.${product}
                    ${CMAKE_SOURCE_DIR}/datasets/model/equity/black_scholes/prices/${folder}/black_scholes_01__${folder}_01__01.json
                    ${CMAKE_SOURCE_DIR}/validation/datasets/price/equity/black_scholes/${folder}/black_scholes_01__${folder}_01__01.json
                    --require-verified
            )
            set_tests_properties(
                validation_black_scholes_${product}
                PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    LABELS "workbench;validation;cached_reference;black_scholes"
                    TIMEOUT 60
            )
        endfunction()
        foreach(product IN ITEMS
            asian_call asian_put
            asset_or_nothing_call asset_or_nothing_put
            athena_autocall cliquet digital_call digital_put
            double_knock_out_call double_knock_out_put
            down_and_in_put down_and_out_put
            european_call european_put
            forward_start_call forward_start_put
            gap_call gap_put
            geometric_asian_call geometric_asian_put
            lookback_option phoenix_autocall phoenix_memory_autocall
            range_accrual straddle up_and_in_call up_and_out_call
        )
            add_cached_black_scholes_validation(${product} ${product}s)
        endforeach()
        add_cached_black_scholes_validation(up_no_touch up_no_touches)
        add_cached_black_scholes_validation(up_one_touch up_one_touches)

        function(add_cached_standalone_rate_validation model product folder)
            add_test(
                NAME validation_${model}_${product}
                COMMAND
                    ${Python3_EXECUTABLE} -m
                    validation.model.fixed_income.${model}.${product}
                    ${CMAKE_SOURCE_DIR}/datasets/model/fixed_income/${model}/prices/${folder}/${model}_01__${folder}_01__01.json
                    ${CMAKE_SOURCE_DIR}/validation/datasets/price/fixed_income/${model}/${folder}/${model}_01__${folder}_01__01.json
                    --require-verified
            )
            set_tests_properties(
                validation_${model}_${product}
                PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    LABELS "workbench;validation;cached_reference;${model}"
                    TIMEOUT 60
            )
        endfunction()
        foreach(model IN ITEMS cir g2 ornstein_uhlenbeck vasicek)
            foreach(product IN ITEMS
                zero_coupon_bond_call zero_coupon_bond_put caplet floorlet
            )
                if(product STREQUAL "zero_coupon_bond_call")
                    set(product_folder zero_coupon_bond_calls)
                elseif(product STREQUAL "zero_coupon_bond_put")
                    set(product_folder zero_coupon_bond_puts)
                elseif(product STREQUAL "caplet")
                    set(product_folder caplets)
                else()
                    set(product_folder floorlets)
                endif()
                add_cached_standalone_rate_validation(
                    ${model} ${product} ${product_folder}
                )
            endforeach()
        endforeach()
        foreach(model IN ITEMS cir ornstein_uhlenbeck vasicek)
            foreach(side IN ITEMS payer receiver)
                add_cached_standalone_rate_validation(
                    ${model}
                    european_${side}_swaption
                    european_${side}_swaptions
                )
            endforeach()
        endforeach()

        function(add_cached_fitted_rate_validation model curve product folder)
            add_test(
                NAME validation_${model}_${curve}_${product}
                COMMAND
                    ${Python3_EXECUTABLE} -m
                    validation.model.fixed_income.${model}.${curve}.${product}
                    ${CMAKE_SOURCE_DIR}/datasets/model/fixed_income/${model}/prices/${curve}/${folder}/${model}_01__${curve}_01__${folder}_01__01.json
                    ${CMAKE_SOURCE_DIR}/validation/datasets/price/fixed_income/${model}/${curve}/${folder}/${model}_01__${curve}_01__${folder}_01__01.json
                    --require-verified
            )
            set_tests_properties(
                validation_${model}_${curve}_${product}
                PROPERTIES
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    LABELS "workbench;validation;cached_reference;${model}"
                    TIMEOUT 60
            )
        endfunction()
        foreach(model IN ITEMS g2_plus_plus hull_white)
            foreach(curve IN ITEMS nelson_siegel svensson)
                foreach(product IN ITEMS
                    zero_coupon_bond_call zero_coupon_bond_put caplet floorlet
                )
                    if(product STREQUAL "zero_coupon_bond_call")
                        set(product_folder zero_coupon_bond_calls)
                    elseif(product STREQUAL "zero_coupon_bond_put")
                        set(product_folder zero_coupon_bond_puts)
                    elseif(product STREQUAL "caplet")
                        set(product_folder caplets)
                    else()
                        set(product_folder floorlets)
                    endif()
                    add_cached_fitted_rate_validation(
                        ${model} ${curve} ${product} ${product_folder}
                    )
                endforeach()
            endforeach()
        endforeach()
        foreach(curve IN ITEMS nelson_siegel svensson)
            foreach(side IN ITEMS payer receiver)
                add_cached_fitted_rate_validation(
                    hull_white
                    ${curve}
                    european_${side}_swaption
                    european_${side}_swaptions
                )
            endforeach()
        endforeach()
    endif()

    # Register one GPU integration test with shared execution rules.
    function(add_cuda_workbench_test name source label timeout)
        set(target test_${name})
        add_executable(${target} EXCLUDE_FROM_ALL ${source})
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        ai_factory_collect_source_dependencies(dependencies ${source})
        target_link_libraries(
            ${target} PRIVATE
            ai_factory_runtime
            ai_factory_dataset_core
            ${dependencies}
        )
        target_compile_features(${target} PRIVATE cxx_std_23)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 23
            CUDA_STANDARD_REQUIRED YES
        )
        add_dependencies(ai_factory_cuda_tests ${target})
        file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" test_source_text)
        set(domain_label common)
        if(test_source_text MATCHES "model/fixed_income/")
            add_dependencies(fixed_income_tests ${target})
            set(domain_label fixed_income)
        endif()
        if(test_source_text MATCHES "model/equity/")
            add_dependencies(equity_tests ${target})
            set(domain_label equity)
        endif()
        if(NOT test_source_text MATCHES "model/(equity|fixed_income)/")
            add_dependencies(common_cuda_tests ${target})
        endif()
        add_test(NAME ${name} COMMAND ${target})
        set_tests_properties(${name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;cuda;${domain_label};${label}"
            RESOURCE_LOCK cuda_gpu
            SKIP_RETURN_CODE 77
            TIMEOUT ${timeout}
        )
    endfunction()

    add_cuda_workbench_test(
        philox_cuda tests/philox_cuda_test.cu philox 30
    )
    add_cuda_workbench_test(
        numerical_robustness_cuda
        tests/numerical_robustness_cuda_test.cu
        numerical_robustness
        30
    )
    add_cuda_workbench_test(
        cuda_pricing_runner
        tests/cuda_pricing_runner_test.cu
        "offline;pricing_runner"
        30
    )
    add_cuda_workbench_test(
        longstaff_schwartz_regressor_cuda
        tests/longstaff_schwartz_regressor_cuda_test.cu
        longstaff_schwartz
        30
    )
    target_link_libraries(
        test_longstaff_schwartz_regressor_cuda PRIVATE
        ai_factory_longstaff_schwartz
    )
    add_cuda_workbench_test(
        bermudan_swaption_cuda
        tests/bermudan_swaption_cuda_test.cpp
        "longstaff_schwartz;bermudan_swaption"
        120
    )
    add_cuda_workbench_test(
        noncentral_chi_square_cuda
        tests/noncentral_chi_square_cuda_test.cu
        distributions
        30
    )
    add_cuda_workbench_test(
        cir_cuda tests/cir_cuda_test.cu cir 30
    )
    add_cuda_workbench_test(
        fixed_income_analytics_contract_cuda
        tests/fixed_income_analytics_contract_cuda_test.cu
        analytics_contract
        30
    )
    add_cuda_workbench_test(
        fixed_income_dynamics_policy_cuda
        tests/fixed_income_dynamics_policy_cuda_test.cu
        dynamics_policy
        30
    )
    add_cuda_workbench_test(
        cir_rate_options_cuda
        tests/cir_rate_options_cuda_test.cu
        "cir;rate_options"
        30
    )
    add_cuda_workbench_test(
        hull_white_cuda tests/hull_white_cuda_test.cu hull_white 30
    )
    add_cuda_workbench_test(
        g2_cuda tests/g2_cuda_test.cu g2 30
    )
    add_cuda_workbench_test(
        g2_caplet_cuda tests/g2_caplet_cuda_test.cu caplet 30
    )
    add_cuda_workbench_test(
        g2_rate_options_cuda
        tests/g2_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_cuda
        tests/g2_plus_plus_cuda_test.cu
        g2_plus_plus
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_caplet_cuda
        tests/g2_plus_plus_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_svensson_caplet_cuda
        tests/g2_plus_plus_svensson_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_rate_options_cuda
        tests/g2_plus_plus_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        hull_white_caplet_cuda
        tests/hull_white_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        hull_white_svensson_caplet_cuda
        tests/hull_white_svensson_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        hull_white_rate_options_cuda
        tests/hull_white_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_caplet_cuda
        tests/ornstein_uhlenbeck_caplet_cuda_test.cu
        caplet
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_rate_options_cuda
        tests/ornstein_uhlenbeck_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_european_swaptions_cuda
        tests/ornstein_uhlenbeck_european_swaptions_cuda_test.cu
        european_swaptions
        30
    )
    add_cuda_workbench_test(
        one_factor_european_swaptions_cuda
        tests/one_factor_european_swaptions_cuda_test.cu
        european_swaptions
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_cuda
        tests/ornstein_uhlenbeck_cuda_test.cu
        ornstein_uhlenbeck
        30
    )
    add_cuda_workbench_test(
        vasicek_cuda tests/vasicek_cuda_test.cu vasicek 30
    )
    add_cuda_workbench_test(
        vasicek_caplet_cuda tests/vasicek_caplet_cuda_test.cu caplet 30
    )
    add_cuda_workbench_test(
        vasicek_rate_options_cuda
        tests/vasicek_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        heston_path_products_cuda
        tests/heston_path_products_cuda_test.cpp
        heston_products
        60
    )
    add_cuda_workbench_test(
        heston_terminal_payoffs_cuda
        tests/heston_terminal_payoffs_cuda_test.cpp
        heston_products
        60
    )
    add_cuda_workbench_test(
        heston_american_option_cuda
        tests/heston_american_option_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        bates_dynamics_cuda
        tests/bates_dynamics_cuda_test.cu
        bates_dynamics
        30
    )
    add_cuda_workbench_test(
        equity_dynamics_policy_cuda
        tests/equity_dynamics_policy_cuda_test.cu
        dynamics_policy
        30
    )
    add_cuda_workbench_test(
        levy_dynamics_cuda
        tests/levy_dynamics_cuda_test.cu
        levy_dynamics
        30
    )
    add_cuda_workbench_test(
        merton_kou_cev_schobel_zhu_dynamics_cuda
        tests/merton_kou_cev_schobel_zhu_dynamics_cuda_test.cu
        equity_dynamics
        30
    )
    add_cuda_workbench_test(
        rough_bergomi_dynamics_cuda
        tests/rough_bergomi_dynamics_cuda_test.cu
        rough_bergomi_dynamics
        30
    )
    add_cuda_workbench_test(
        sabr_absorbing_boundary_cuda
        tests/sabr_absorbing_boundary_cuda_test.cpp
        "sabr;absorbing_boundary"
        60
    )
    add_cuda_workbench_test(
        volterra_kernel_policy_cuda
        tests/volterra_kernel_policy_cuda_test.cu
        "volterra;kernel_policy"
        30
    )
    if(AI_FACTORY_MATHDX_ROOT)
        add_cuda_workbench_test(
            rough_bergomi_european_option_cuda
            tests/rough_bergomi_european_option_cuda_test.cpp
            "rough_bergomi_products;rough_sabr_products"
            120
        )
        add_cuda_workbench_test(
            rough_volterra_product_policy_cuda
            tests/rough_volterra_product_policy_cuda_test.cu
            "rough_bergomi_products;product_policy"
            120
        )
        add_cuda_workbench_test(
            rough_volterra_samples_cuda
            tests/rough_volterra_samples_cuda_test.cpp
            "samples;rough_bergomi;rough_sabr"
            120
        )
        target_include_directories(
            test_rough_volterra_product_policy_cuda PRIVATE
            ${AI_FACTORY_MATHDX_ROOT}/include
            ${AI_FACTORY_MATHDX_ROOT}/external/cutlass/include
        )
    endif()
    add_cuda_workbench_test(
        rough_heston_european_option_cuda
        tests/rough_heston_european_option_cuda_test.cu
        rough_heston_products
        120
    )
    add_cuda_workbench_test(
        rough_heston_samples_cuda
        tests/rough_heston_samples_cuda_test.cu
        "samples;rough_heston"
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_samples_cuda
        tests/quadratic_rough_heston_samples_cuda_test.cu
        "samples;quadratic_rough_heston;numerical_robustness"
        120
    )
    add_cuda_workbench_test(
        black_scholes_cuda
        tests/black_scholes_cuda_test.cu
        black_scholes
        30
    )
    add_cuda_workbench_test(
        black_scholes_samples_cuda
        tests/black_scholes_samples_cuda_test.cu
        samples
        30
    )
    add_cuda_workbench_test(
        bates_path_products_cuda
        tests/bates_path_products_cuda_test.cpp
        bates_products
        60
    )
    add_cuda_workbench_test(
        bates_terminal_payoffs_cuda
        tests/bates_terminal_payoffs_cuda_test.cpp
        bates_products
        60
    )
    add_cuda_workbench_test(
        bates_american_option_cuda
        tests/bates_american_option_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        levy_american_option_cuda
        tests/levy_american_option_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        additional_american_options_cuda
        tests/additional_american_options_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        cuda_kernel_diagnostics
        tests/cuda_kernel_diagnostics_test.cu
        kernel_diagnostics
        30
    )
    add_cuda_workbench_test(
        policy_size_budgets_cuda
        tests/policy_size_budgets_cuda_test.cu
        "kernel_diagnostics;policy_budgets"
        30
    )

endif()
