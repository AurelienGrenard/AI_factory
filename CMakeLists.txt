cmake_minimum_required(VERSION 3.18)

project(ai_factory_cuda_workbench LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(nlohmann_json 3.2.0 REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Keep the architecture list overridable from the CMake command line. CUDA
# 11.8 introduced native Ada (SM 8.9) support; older toolkits retain the
# portable Volta/Ampere default instead of failing configuration.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.8)
    set(_cuda_workbench_default_architectures "70;86;89")
else()
    set(_cuda_workbench_default_architectures "70;86")
    message(STATUS
        "CUDA < 11.8: SM 8.9 is unavailable; using architectures 70 and 86"
    )
endif()
set(
    CUDA_WORKBENCH_ARCHITECTURES "${_cuda_workbench_default_architectures}"
    CACHE STRING
    "CUDA architectures (for example 89, 86, or 70;86;89)"
)
set(CMAKE_CUDA_ARCHITECTURES "${CUDA_WORKBENCH_ARCHITECTURES}")

option(
    CUDA_WORKBENCH_FAST_MATH
    "Use CUDA fast FP32 transcendental functions in pricing kernels"
    OFF
)

# Host-only generation and serialization helpers shared by all generators.
add_library(ai_factory_dataset_tools STATIC
    tools/datasets/dataset.cpp
    tools/datasets/nelson_siegel_generation.cpp
    tools/datasets/ornstein_uhlenbeck_generation.cpp
)
target_include_directories(ai_factory_dataset_tools PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    ai_factory_dataset_tools PUBLIC nlohmann_json::nlohmann_json
)
target_compile_features(ai_factory_dataset_tools PUBLIC cxx_std_17)
target_compile_options(ai_factory_dataset_tools PRIVATE -O3)

add_library(cuda_workbench STATIC
    src/curve/nelson_siegel/dataset.cpp
    src/model/heston/dataset.cpp
    src/model/heston/american_put.cu
    src/model/heston/asian_call.cu
    src/model/heston/european_call.cu
    src/model/heston/lookback_option.cu
    src/model/hull_white/dataset.cpp
    src/model/hull_white/nelson_siegel/caplet.cu
    src/model/ornstein_uhlenbeck/caplet.cu
    src/model/ornstein_uhlenbeck/dataset.cpp
    src/product/american_put/dataset.cpp
    src/product/asian_call/dataset.cpp
    src/product/caplet/dataset.cpp
    src/product/european_call/dataset.cpp
    src/product/lookback_option/dataset.cpp
)
target_include_directories(cuda_workbench PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cuda_workbench PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(cuda_workbench PUBLIC cxx_std_17)
set_target_properties(cuda_workbench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_compile_options(cuda_workbench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
if(CUDA_WORKBENCH_FAST_MATH)
    target_compile_options(cuda_workbench PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

# Register one host-only parameter generator with shared build settings.
function(add_parameter_generator target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ai_factory_dataset_tools)
    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

# Register one CUDA price generator with shared build settings.
function(add_price_generator target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(
        ${target} PRIVATE cuda_workbench ai_factory_dataset_tools
    )
    target_compile_features(${target} PRIVATE cxx_std_17)
endfunction()

add_price_generator(
    generate_heston_european_calls_01
    catalog/price/heston/european_calls/heston_01__european_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_european_calls_02
    catalog/price/heston/european_calls/heston_01__european_calls_01__02/generator.cpp
)
add_price_generator(
    generate_heston_asian_calls_01
    catalog/price/heston/asian_calls/heston_01__asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_lookback_options_01
    catalog/price/heston/lookback_options/heston_01__lookback_options_01__01/generator.cpp
)
add_price_generator(
    generate_heston_american_puts_01
    catalog/price/heston/american_puts/heston_01__american_puts_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_caplets_01
    catalog/price/hull_white/nelson_siegel/caplets/hull_white_01__nelson_siegel_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_caplets_01
    catalog/price/ornstein_uhlenbeck/caplets/ornstein_uhlenbeck_01__caplets_01__01/generator.cpp
)

add_parameter_generator(
    generate_heston_01 catalog/model/heston/heston_01/generator.cpp
)
add_parameter_generator(
    generate_hull_white_01 catalog/model/hull_white/hull_white_01/generator.cpp
)
add_parameter_generator(
    generate_ornstein_uhlenbeck_01
    catalog/model/ornstein_uhlenbeck/ornstein_uhlenbeck_01/generator.cpp
)
add_parameter_generator(
    generate_nelson_siegel_01
    catalog/curve/nelson_siegel/nelson_siegel_01/generator.cpp
)
add_parameter_generator(
    generate_european_calls_01
    catalog/product/equity/european_calls/european_calls_01/generator.cpp
)
add_parameter_generator(
    generate_asian_calls_01
    catalog/product/equity/asian_calls/asian_calls_01/generator.cpp
)
add_parameter_generator(
    generate_lookback_options_01
    catalog/product/equity/lookback_options/lookback_options_01/generator.cpp
)
add_parameter_generator(
    generate_american_puts_01
    catalog/product/equity/american_puts/american_puts_01/generator.cpp
)
add_parameter_generator(
    generate_caplets_01
    catalog/product/fixed_income/caplets/caplets_01/generator.cpp
)

if(BUILD_TESTING)
    add_executable(
        test_dataset_catalog
        tests/catalog_contract_test.cpp
    )
    target_include_directories(test_dataset_catalog PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(
        test_dataset_catalog
        PRIVATE cuda_workbench ai_factory_dataset_tools
    )
    target_compile_features(test_dataset_catalog PRIVATE cxx_std_17)
    add_test(
        NAME dataset_catalog
        COMMAND test_dataset_catalog
    )
    set_tests_properties(dataset_catalog PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;catalog"
    )

    # Register one GPU integration test with shared execution rules.
    function(add_cuda_workbench_test name source label timeout)
        set(target test_${name})
        add_executable(${target} ${source})
        target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(
            ${target} PRIVATE cuda_workbench ai_factory_dataset_tools
        )
        target_compile_features(${target} PRIVATE cxx_std_17)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED YES
        )
        add_test(NAME ${name} COMMAND ${target})
        set_tests_properties(${name} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;cuda;${label}"
            RESOURCE_LOCK cuda_gpu
            SKIP_RETURN_CODE 77
            TIMEOUT ${timeout}
        )
    endfunction()

    add_cuda_workbench_test(
        hull_white_cuda tests/hull_white_cuda_test.cu hull_white 30
    )
    add_cuda_workbench_test(
        hull_white_caplet_cuda
        tests/hull_white_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_caplet_cuda
        tests/ornstein_uhlenbeck_caplet_cuda_test.cu
        caplet
        30
    )
    add_cuda_workbench_test(
        heston_path_products_cuda
        tests/heston_path_products_cuda_test.cpp
        heston_products
        60
    )
    add_cuda_workbench_test(
        heston_american_put_cuda
        tests/heston_american_put_cuda_test.cpp
        american_put
        120
    )

    # Build explicitly when comparing American-put launch configurations.
    add_executable(
        heston_american_put_benchmark EXCLUDE_FROM_ALL
        tests/performance/heston_american_put_benchmark.cpp
    )
    target_include_directories(heston_american_put_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(
        heston_american_put_benchmark
        PRIVATE cuda_workbench
    )
    target_compile_features(
        heston_american_put_benchmark PRIVATE cxx_std_17
    )
endif()
