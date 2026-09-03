# Configure the CUDA workbench, generated ownership graph, tests and validation targets.
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


# Domain modules own target registration; this root only configures and orchestrates.
include(cmake/AIFactoryRuntime.cmake)
include(cmake/AIFactoryTargets.cmake)
include(cmake/AIFactoryCatalog.cmake)
include(cmake/AIFactoryPerformance.cmake)
include(cmake/AIFactoryTests.cmake)
