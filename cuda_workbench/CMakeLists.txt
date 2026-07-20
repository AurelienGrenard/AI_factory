cmake_minimum_required(VERSION 3.18)

project(ai_factory_cuda_workbench LANGUAGES CXX CUDA)

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

# Small host-only library used by every model/product registry generator.
add_library(cuda_workbench_registry STATIC
    tools/registry/parameter_database.cpp
    tools/registry/result_output.cpp
)
target_include_directories(cuda_workbench_registry PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    cuda_workbench_registry PUBLIC nlohmann_json::nlohmann_json
)
target_compile_features(cuda_workbench_registry PUBLIC cxx_std_17)
target_compile_options(cuda_workbench_registry PRIVATE -O3)

add_library(cuda_workbench STATIC
    src/heston/common.cpp
    src/heston/european_call.cu
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

add_executable(
    generate_heston_european_calls_01
    registry/production/results/heston/european_calls/generators/heston_01__european_calls_01__cpp_gpu_philox_01.cpp
)
target_include_directories(generate_heston_european_calls_01 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    generate_heston_european_calls_01
    PRIVATE cuda_workbench cuda_workbench_registry
)
target_compile_features(generate_heston_european_calls_01 PRIVATE cxx_std_17)

# These two host executables create the local model and product JSON/YAML.
add_executable(
    generate_heston_01
    registry/production/models/heston/generators/heston_01.cpp
)
target_link_libraries(generate_heston_01 PRIVATE cuda_workbench_registry)

add_executable(
    generate_european_calls_01
    registry/production/products/european_calls/generators/european_calls_01.cpp
)
target_link_libraries(
    generate_european_calls_01 PRIVATE cuda_workbench_registry
)
