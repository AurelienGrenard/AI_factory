cmake_minimum_required(VERSION 3.18)

project(ai_factory_cpp LANGUAGES CXX CUDA)

find_package(OpenMP)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(
    AI_FACTORY_CUDA_ARCHITECTURES "70;86"
    CACHE STRING
    "CUDA architectures for the production library (V100 and recent NVIDIA GPUs)"
)
set(
    CMAKE_CUDA_ARCHITECTURES "${AI_FACTORY_CUDA_ARCHITECTURES}"
    CACHE STRING "CUDA architectures" FORCE
)

add_library(ai_factory_cpp
    ai_factory/cpu/common/philox.cpp
    ai_factory/cpu/black_76/caplets.cpp
    ai_factory/cpu/black_76/swaptions.cpp
    ai_factory/cpu/hull_white/caplets.cpp
    ai_factory/cpu/hull_white/interest_rate_swaps.cpp
    ai_factory/cpu/hull_white/zero_coupon_bonds.cpp
    ai_factory/cpu/hull_white/swaptions.cpp
    ai_factory/cpu/hull_white/bermudan_swaptions.cpp
    ai_factory/cpu/cir/interest_rate_swaps.cpp
    ai_factory/cpu/cir/zero_coupon_bonds.cpp
    ai_factory/cpu/cir/swaptions.cpp
    ai_factory/cpu/cir/bermudan_swaptions.cpp
    ai_factory/cpu/cir/caplets.cpp
    ai_factory/cpu/cir_plus_plus/interest_rate_swaps.cpp
    ai_factory/cpu/cir_plus_plus/zero_coupon_bonds.cpp
    ai_factory/cpu/cir_plus_plus/swaptions.cpp
    ai_factory/cpu/cir_plus_plus/bermudan_swaptions.cpp
    ai_factory/cpu/cir_plus_plus/caplets.cpp
    ai_factory/cpu/g2_plus_plus/interest_rate_swaps.cpp
    ai_factory/cpu/g2_plus_plus/zero_coupon_bonds.cpp
    ai_factory/cpu/g2_plus_plus/swaptions.cpp
    ai_factory/cpu/g2_plus_plus/bermudan_swaptions.cpp
    ai_factory/cpu/g2_plus_plus/caplets.cpp
    ai_factory/cpu/black_scholes/common.cpp
    ai_factory/cpu/black_scholes/autocalls.cpp
    ai_factory/cpu/black_scholes/american_puts.cpp
    ai_factory/cpu/black_scholes/european_calls.cpp
    ai_factory/cpu/black_scholes/digital_calls.cpp
    ai_factory/cpu/black_scholes/asian_arithmetic_calls.cpp
    ai_factory/cpu/black_scholes/lookback_fixed_calls.cpp
    ai_factory/cpu/black_scholes/volatility_swaps.cpp
    ai_factory/cpu/black_scholes/down_and_out_calls.cpp
    ai_factory/cpu/black_scholes/down_and_in_calls.cpp
    ai_factory/cpu/black_scholes/up_and_out_calls.cpp
    ai_factory/cpu/black_scholes/up_and_in_calls.cpp
    ai_factory/cpu/heston/common.cpp
    ai_factory/cpu/heston/autocalls.cpp
    ai_factory/cpu/heston/american_puts.cpp
    ai_factory/cpu/heston/european_calls.cpp
    ai_factory/cpu/heston/digital_calls.cpp
    ai_factory/cpu/heston/asian_arithmetic_calls.cpp
    ai_factory/cpu/heston/lookback_fixed_calls.cpp
    ai_factory/cpu/heston/volatility_swaps.cpp
    ai_factory/cpu/heston/down_and_out_calls.cpp
    ai_factory/cpu/heston/down_and_in_calls.cpp
    ai_factory/cpu/heston/up_and_out_calls.cpp
    ai_factory/cpu/heston/up_and_in_calls.cpp
    ai_factory/cpu/rough_bergomi/common.cpp
    ai_factory/cpu/rough_bergomi/autocalls.cpp
    ai_factory/cpu/rough_bergomi/asian_arithmetic_calls.cpp
    ai_factory/cpu/rough_bergomi/european_calls.cpp
    ai_factory/cpu/rough_bergomi/digital_calls.cpp
    ai_factory/cpu/rough_bergomi/lookback_fixed_calls.cpp
    ai_factory/cpu/rough_bergomi/volatility_swaps.cpp
    ai_factory/cpu/rough_bergomi/down_and_out_calls.cpp
    ai_factory/cpu/rough_bergomi/down_and_in_calls.cpp
    ai_factory/cpu/rough_bergomi/up_and_out_calls.cpp
    ai_factory/cpu/rough_bergomi/up_and_in_calls.cpp
    ai_factory/cpu/rough_heston/common.cpp
    ai_factory/cpu/rough_heston/american_puts.cpp
    ai_factory/cpu/rough_heston/autocalls.cpp
    ai_factory/cpu/rough_heston/asian_arithmetic_calls.cpp
    ai_factory/cpu/rough_heston/european_calls.cpp
    ai_factory/cpu/rough_heston/digital_calls.cpp
    ai_factory/cpu/rough_heston/lookback_fixed_calls.cpp
    ai_factory/cpu/rough_heston/volatility_swaps.cpp
    ai_factory/cpu/rough_heston/down_and_out_calls.cpp
    ai_factory/cpu/rough_heston/down_and_in_calls.cpp
    ai_factory/cpu/rough_heston/up_and_out_calls.cpp
    ai_factory/cpu/rough_heston/up_and_in_calls.cpp
)

target_include_directories(ai_factory_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ai_factory_cpp PUBLIC cxx_std_20)

add_library(ai_factory_cuda_shared SHARED
    ai_factory/cpu/common/philox.cpp
    ai_factory/cpu/black_76/caplets.cpp
    ai_factory/cpu/black_76/swaptions.cpp
    ai_factory/cpu/hull_white/caplets.cpp
    ai_factory/cpu/hull_white/interest_rate_swaps.cpp
    ai_factory/cpu/hull_white/zero_coupon_bonds.cpp
    ai_factory/cpu/hull_white/swaptions.cpp
    ai_factory/cpu/hull_white/bermudan_swaptions.cpp
    ai_factory/cpu/cir/interest_rate_swaps.cpp
    ai_factory/cpu/cir/zero_coupon_bonds.cpp
    ai_factory/cpu/cir/swaptions.cpp
    ai_factory/cpu/cir/bermudan_swaptions.cpp
    ai_factory/cpu/cir/caplets.cpp
    ai_factory/cpu/cir_plus_plus/interest_rate_swaps.cpp
    ai_factory/cpu/cir_plus_plus/zero_coupon_bonds.cpp
    ai_factory/cpu/cir_plus_plus/swaptions.cpp
    ai_factory/cpu/cir_plus_plus/bermudan_swaptions.cpp
    ai_factory/cpu/cir_plus_plus/caplets.cpp
    ai_factory/cpu/g2_plus_plus/interest_rate_swaps.cpp
    ai_factory/cpu/g2_plus_plus/zero_coupon_bonds.cpp
    ai_factory/cpu/g2_plus_plus/swaptions.cpp
    ai_factory/cpu/g2_plus_plus/bermudan_swaptions.cpp
    ai_factory/cpu/g2_plus_plus/caplets.cpp
    ai_factory/cpu/black_scholes/common.cpp
    ai_factory/cpu/black_scholes/autocalls.cpp
    ai_factory/cpu/black_scholes/american_puts.cpp
    ai_factory/cpu/black_scholes/european_calls.cpp
    ai_factory/cpu/black_scholes/digital_calls.cpp
    ai_factory/cpu/black_scholes/asian_arithmetic_calls.cpp
    ai_factory/cpu/black_scholes/lookback_fixed_calls.cpp
    ai_factory/cpu/black_scholes/volatility_swaps.cpp
    ai_factory/cpu/black_scholes/down_and_out_calls.cpp
    ai_factory/cpu/black_scholes/down_and_in_calls.cpp
    ai_factory/cpu/black_scholes/up_and_out_calls.cpp
    ai_factory/cpu/black_scholes/up_and_in_calls.cpp
    ai_factory/cpu/heston/common.cpp
    ai_factory/cpu/heston/autocalls.cpp
    ai_factory/cpu/heston/american_puts.cpp
    ai_factory/cpu/heston/european_calls.cpp
    ai_factory/cpu/heston/digital_calls.cpp
    ai_factory/cpu/heston/asian_arithmetic_calls.cpp
    ai_factory/cpu/heston/lookback_fixed_calls.cpp
    ai_factory/cpu/heston/volatility_swaps.cpp
    ai_factory/cpu/heston/down_and_out_calls.cpp
    ai_factory/cpu/heston/down_and_in_calls.cpp
    ai_factory/cpu/heston/up_and_out_calls.cpp
    ai_factory/cpu/heston/up_and_in_calls.cpp
    ai_factory/cpu/rough_bergomi/common.cpp
    ai_factory/cpu/rough_bergomi/autocalls.cpp
    ai_factory/cpu/rough_bergomi/asian_arithmetic_calls.cpp
    ai_factory/cpu/rough_bergomi/european_calls.cpp
    ai_factory/cpu/rough_bergomi/digital_calls.cpp
    ai_factory/cpu/rough_bergomi/lookback_fixed_calls.cpp
    ai_factory/cpu/rough_bergomi/volatility_swaps.cpp
    ai_factory/cpu/rough_bergomi/down_and_out_calls.cpp
    ai_factory/cpu/rough_bergomi/down_and_in_calls.cpp
    ai_factory/cpu/rough_bergomi/up_and_out_calls.cpp
    ai_factory/cpu/rough_bergomi/up_and_in_calls.cpp
    ai_factory/cpu/rough_heston/common.cpp
    ai_factory/cpu/rough_heston/american_puts.cpp
    ai_factory/cpu/rough_heston/autocalls.cpp
    ai_factory/cpu/rough_heston/asian_arithmetic_calls.cpp
    ai_factory/cpu/rough_heston/european_calls.cpp
    ai_factory/cpu/rough_heston/digital_calls.cpp
    ai_factory/cpu/rough_heston/lookback_fixed_calls.cpp
    ai_factory/cpu/rough_heston/volatility_swaps.cpp
    ai_factory/cpu/rough_heston/down_and_out_calls.cpp
    ai_factory/cpu/rough_heston/down_and_in_calls.cpp
    ai_factory/cpu/rough_heston/up_and_out_calls.cpp
    ai_factory/cpu/rough_heston/up_and_in_calls.cpp
    ai_factory/cuda/heston/lookback_fixed_calls.cu
    ai_factory/cuda/heston/european_calls.cu
    ai_factory/cuda/heston/digital_calls.cu
    ai_factory/cuda/heston/autocalls.cu
    ai_factory/cuda/heston/american_puts.cu
    ai_factory/cuda/heston/asian_arithmetic_calls.cu
    ai_factory/cuda/heston/volatility_swaps.cu
    ai_factory/cuda/heston/down_and_out_calls.cu
    ai_factory/cuda/heston/down_and_in_calls.cu
    ai_factory/cuda/heston/up_and_out_calls.cu
    ai_factory/cuda/heston/up_and_in_calls.cu
    ai_factory/cuda/heston/paths.cu
    ai_factory/cuda/black_scholes/american_puts.cu
    ai_factory/cuda/black_scholes/european_calls.cu
    ai_factory/cuda/black_scholes/digital_calls.cu
    ai_factory/cuda/black_scholes/autocalls.cu
    ai_factory/cuda/black_scholes/asian_arithmetic_calls.cu
    ai_factory/cuda/black_scholes/lookback_fixed_calls.cu
    ai_factory/cuda/black_scholes/paths.cu
    ai_factory/cuda/black_scholes/volatility_swaps.cu
    ai_factory/cuda/black_scholes/down_and_out_calls.cu
    ai_factory/cuda/black_scholes/down_and_in_calls.cu
    ai_factory/cuda/black_scholes/up_and_out_calls.cu
    ai_factory/cuda/black_scholes/up_and_in_calls.cu
    ai_factory/cuda/rough_bergomi/asian_arithmetic_calls.cu
    ai_factory/cuda/rough_bergomi/european_calls.cu
    ai_factory/cuda/rough_bergomi/digital_calls.cu
    ai_factory/cuda/rough_bergomi/autocalls.cu
    ai_factory/cuda/rough_bergomi/lookback_fixed_calls.cu
    ai_factory/cuda/rough_bergomi/volatility_swaps.cu
    ai_factory/cuda/rough_bergomi/down_and_out_calls.cu
    ai_factory/cuda/rough_bergomi/down_and_in_calls.cu
    ai_factory/cuda/rough_bergomi/up_and_out_calls.cu
    ai_factory/cuda/rough_bergomi/up_and_in_calls.cu
    ai_factory/cuda/rough_bergomi/paths.cu
    ai_factory/cuda/rough_heston/asian_arithmetic_calls.cu
    ai_factory/cuda/rough_heston/american_puts.cu
    ai_factory/cuda/rough_heston/paths.cu
    ai_factory/cuda/rough_heston/european_calls.cu
    ai_factory/cuda/rough_heston/digital_calls.cu
    ai_factory/cuda/rough_heston/autocalls.cu
    ai_factory/cuda/rough_heston/lookback_fixed_calls.cu
    ai_factory/cuda/rough_heston/volatility_swaps.cu
    ai_factory/cuda/rough_heston/down_and_out_calls.cu
    ai_factory/cuda/rough_heston/down_and_in_calls.cu
    ai_factory/cuda/rough_heston/up_and_out_calls.cu
    ai_factory/cuda/rough_heston/up_and_in_calls.cu
    ai_factory/cuda/black_76/caplets.cu
    ai_factory/cuda/black_76/swaptions.cu
    ai_factory/cuda/hull_white/caplets.cu
    ai_factory/cuda/hull_white/interest_rate_swaps.cu
    ai_factory/cuda/hull_white/zero_coupon_bonds.cu
    ai_factory/cuda/hull_white/swaptions.cu
    ai_factory/cuda/hull_white/bermudan_swaptions.cu
    ai_factory/cuda/cir/interest_rate_swaps.cu
    ai_factory/cuda/cir/zero_coupon_bonds.cu
    ai_factory/cuda/cir/swaptions.cu
    ai_factory/cuda/cir/bermudan_swaptions.cu
    ai_factory/cuda/cir/caplets.cu
    ai_factory/cuda/cir_plus_plus/interest_rate_swaps.cu
    ai_factory/cuda/cir_plus_plus/zero_coupon_bonds.cu
    ai_factory/cuda/cir_plus_plus/swaptions.cu
    ai_factory/cuda/cir_plus_plus/bermudan_swaptions.cu
    ai_factory/cuda/cir_plus_plus/caplets.cu
    ai_factory/cuda/g2_plus_plus/interest_rate_swaps.cu
    ai_factory/cuda/g2_plus_plus/zero_coupon_bonds.cu
    ai_factory/cuda/g2_plus_plus/swaptions.cu
    ai_factory/cuda/g2_plus_plus/bermudan_swaptions.cu
    ai_factory/cuda/g2_plus_plus/caplets.cu
    ai_factory/c_api/c_api.cpp
)
set_target_properties(ai_factory_cuda_shared PROPERTIES
    OUTPUT_NAME ai_factory_cuda
)
target_include_directories(ai_factory_cuda_shared PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ai_factory_cuda_shared PRIVATE cxx_std_17)
target_compile_options(ai_factory_cuda_shared PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ai_factory_cpp PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(ai_factory_cuda_shared PRIVATE OpenMP::OpenMP_CXX)
endif()
