cmake_minimum_required(VERSION 3.18)

project(ai_factory_cuda_workbench LANGUAGES CXX CUDA)

include(CTest)

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
    src/heston/parameters.cpp
    src/heston/american_put.cu
    src/heston/european_call.cu
    src/products/american_put.cpp
    src/products/european_call.cpp
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

add_executable(
    generate_heston_european_calls_01
    generators/price/heston/european_calls/heston_01__european_calls_01__01.cpp
)
target_include_directories(generate_heston_european_calls_01 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    generate_heston_european_calls_01
    PRIVATE cuda_workbench ai_factory_dataset_tools
)
target_compile_features(generate_heston_european_calls_01 PRIVATE cxx_std_17)

add_executable(
    generate_heston_european_calls_02
    generators/price/heston/european_calls/heston_01__european_calls_01__02.cpp
)
target_include_directories(generate_heston_european_calls_02 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    generate_heston_european_calls_02
    PRIVATE cuda_workbench ai_factory_dataset_tools
)
target_compile_features(
    generate_heston_european_calls_02 PRIVATE cxx_std_17
)

# These host executables create datasets, previews, and catalog YAML.
add_executable(
    generate_heston_01
    generators/model/heston/heston_01.cpp
)
target_link_libraries(generate_heston_01 PRIVATE ai_factory_dataset_tools)

add_executable(
    generate_european_calls_01
    generators/product/european_calls/european_calls_01.cpp
)
target_link_libraries(
    generate_european_calls_01 PRIVATE ai_factory_dataset_tools
)

add_executable(
    generate_american_puts_01
    generators/product/american_puts/american_puts_01.cpp
)
target_link_libraries(
    generate_american_puts_01 PRIVATE ai_factory_dataset_tools
)

add_executable(
    generate_heston_american_puts_01
    generators/price/heston/american_puts/heston_01__american_puts_01__01.cpp
)
target_include_directories(generate_heston_american_puts_01 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    generate_heston_american_puts_01
    PRIVATE cuda_workbench ai_factory_dataset_tools
)
target_compile_features(generate_heston_american_puts_01 PRIVATE cxx_std_17)

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

    add_executable(
        test_heston_american_put_cuda
        tests/heston_american_put_cuda_test.cpp
    )
    target_include_directories(test_heston_american_put_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(
        test_heston_american_put_cuda
        PRIVATE cuda_workbench ai_factory_dataset_tools
    )
    target_compile_features(test_heston_american_put_cuda PRIVATE cxx_std_17)
    add_test(
        NAME heston_american_put_cuda
        COMMAND test_heston_american_put_cuda
    )
    set_tests_properties(heston_american_put_cuda PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;cuda;american_put"
        RESOURCE_LOCK cuda_gpu
        SKIP_RETURN_CODE 77
        TIMEOUT 120
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
