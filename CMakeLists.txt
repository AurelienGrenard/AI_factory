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

# Host-only generation and serialization helpers shared by all generators.
add_library(ai_factory_dataset_tools STATIC
    tools/datasets/autocall_generation.cpp
    tools/datasets/cliquet_generation.cpp
    tools/datasets/range_accrual_generation.cpp
    tools/datasets/dataset.cpp
    tools/datasets/g2_generation.cpp
    tools/datasets/nelson_siegel_generation.cpp
    tools/datasets/ornstein_uhlenbeck_generation.cpp
    tools/datasets/svensson_generation.cpp
    tools/datasets/vasicek_generation.cpp
    tools/datasets/dataset_validation.cpp
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
    src/common/cuda_kernel_diagnostics.cpp
    src/common/longstaff_schwartz/exercise_schedule.cu
    src/common/longstaff_schwartz/launch.cu
    src/common/longstaff_schwartz/workspace.cu
    src/curve/nelson_siegel/dataset.cpp
    src/curve/svensson/dataset.cpp
    src/model/fixed_income/g2/dataset.cpp
    src/model/fixed_income/g2/rate_option.cu
    src/model/fixed_income/g2/zero_coupon_bond_option.cu
    src/model/fixed_income/g2_plus_plus/dataset.cpp
    src/model/fixed_income/g2_plus_plus/nelson_siegel/rate_option.cu
    src/model/fixed_income/g2_plus_plus/nelson_siegel/zero_coupon_bond_option.cu
    src/model/fixed_income/g2_plus_plus/svensson/rate_option.cu
    src/model/fixed_income/g2_plus_plus/svensson/zero_coupon_bond_option.cu
    src/model/equity/heston/dataset.cpp
    src/model/equity/heston/american_option.cu
    src/model/equity/heston/athena_autocall.cu
    src/model/equity/heston/asian_option.cu
    src/model/equity/heston/asset_or_nothing_option.cu
    src/model/equity/heston/cliquet.cu
    src/model/equity/heston/range_accrual.cu
    src/model/equity/heston/digital_option.cu
    src/model/equity/heston/double_knock_out_option.cu
    src/model/equity/heston/down_and_in_option.cu
    src/model/equity/heston/down_and_out_option.cu
    src/model/equity/heston/european_option.cu
    src/model/equity/heston/gap_option.cu
    src/model/equity/heston/forward_start_option.cu
    src/model/equity/heston/geometric_asian_option.cu
    src/model/equity/heston/lookback_option.cu
    src/model/equity/heston/phoenix_autocall.cu
    src/model/equity/heston/phoenix_memory_autocall.cu
    src/model/equity/heston/straddle.cu
    src/model/equity/heston/up_and_in_option.cu
    src/model/equity/heston/up_no_touch.cu
    src/model/equity/heston/up_one_touch.cu
    src/model/equity/heston/up_and_out_option.cu
    src/model/equity/bates/dataset.cpp
    src/model/equity/bates/american_option.cu
    src/model/equity/bates/athena_autocall.cu
    src/model/equity/bates/asian_option.cu
    src/model/equity/bates/asset_or_nothing_option.cu
    src/model/equity/bates/cliquet.cu
    src/model/equity/bates/range_accrual.cu
    src/model/equity/bates/digital_option.cu
    src/model/equity/bates/double_knock_out_option.cu
    src/model/equity/bates/down_and_in_option.cu
    src/model/equity/bates/down_and_out_option.cu
    src/model/equity/bates/european_option.cu
    src/model/equity/bates/gap_option.cu
    src/model/equity/bates/forward_start_option.cu
    src/model/equity/bates/geometric_asian_option.cu
    src/model/equity/bates/lookback_option.cu
    src/model/equity/bates/phoenix_autocall.cu
    src/model/equity/bates/phoenix_memory_autocall.cu
    src/model/equity/bates/straddle.cu
    src/model/equity/bates/up_and_in_option.cu
    src/model/equity/bates/up_no_touch.cu
    src/model/equity/bates/up_one_touch.cu
    src/model/equity/bates/up_and_out_option.cu
    src/model/equity/variance_gamma/dataset.cpp
    src/model/equity/variance_gamma/american_option.cu
    src/model/equity/variance_gamma/asian_option.cu
    src/model/equity/variance_gamma/asset_or_nothing_option.cu
    src/model/equity/variance_gamma/athena_autocall.cu
    src/model/equity/variance_gamma/cliquet.cu
    src/model/equity/variance_gamma/digital_option.cu
    src/model/equity/variance_gamma/double_knock_out_option.cu
    src/model/equity/variance_gamma/down_and_in_option.cu
    src/model/equity/variance_gamma/down_and_out_option.cu
    src/model/equity/variance_gamma/european_option.cu
    src/model/equity/variance_gamma/forward_start_option.cu
    src/model/equity/variance_gamma/gap_option.cu
    src/model/equity/variance_gamma/geometric_asian_option.cu
    src/model/equity/variance_gamma/lookback_option.cu
    src/model/equity/variance_gamma/phoenix_autocall.cu
    src/model/equity/variance_gamma/phoenix_memory_autocall.cu
    src/model/equity/variance_gamma/range_accrual.cu
    src/model/equity/variance_gamma/straddle.cu
    src/model/equity/variance_gamma/up_and_in_option.cu
    src/model/equity/variance_gamma/up_and_out_option.cu
    src/model/equity/variance_gamma/up_no_touch.cu
    src/model/equity/variance_gamma/up_one_touch.cu
    src/model/equity/normal_inverse_gaussian/dataset.cpp
    src/model/equity/normal_inverse_gaussian/american_option.cu
    src/model/equity/normal_inverse_gaussian/asian_option.cu
    src/model/equity/normal_inverse_gaussian/asset_or_nothing_option.cu
    src/model/equity/normal_inverse_gaussian/athena_autocall.cu
    src/model/equity/normal_inverse_gaussian/cliquet.cu
    src/model/equity/normal_inverse_gaussian/digital_option.cu
    src/model/equity/normal_inverse_gaussian/double_knock_out_option.cu
    src/model/equity/normal_inverse_gaussian/down_and_in_option.cu
    src/model/equity/normal_inverse_gaussian/down_and_out_option.cu
    src/model/equity/normal_inverse_gaussian/european_option.cu
    src/model/equity/normal_inverse_gaussian/forward_start_option.cu
    src/model/equity/normal_inverse_gaussian/gap_option.cu
    src/model/equity/normal_inverse_gaussian/geometric_asian_option.cu
    src/model/equity/normal_inverse_gaussian/lookback_option.cu
    src/model/equity/normal_inverse_gaussian/phoenix_autocall.cu
    src/model/equity/normal_inverse_gaussian/phoenix_memory_autocall.cu
    src/model/equity/normal_inverse_gaussian/range_accrual.cu
    src/model/equity/normal_inverse_gaussian/straddle.cu
    src/model/equity/normal_inverse_gaussian/up_and_in_option.cu
    src/model/equity/normal_inverse_gaussian/up_and_out_option.cu
    src/model/equity/normal_inverse_gaussian/up_no_touch.cu
    src/model/equity/normal_inverse_gaussian/up_one_touch.cu
    src/model/fixed_income/hull_white/dataset.cpp
    src/model/fixed_income/hull_white/nelson_siegel/rate_option.cu
    src/model/fixed_income/hull_white/nelson_siegel/zero_coupon_bond_option.cu
    src/model/fixed_income/hull_white/svensson/rate_option.cu
    src/model/fixed_income/hull_white/svensson/zero_coupon_bond_option.cu
    src/model/fixed_income/ornstein_uhlenbeck/rate_option.cu
    src/model/fixed_income/ornstein_uhlenbeck/zero_coupon_bond_option.cu
    src/model/fixed_income/ornstein_uhlenbeck/dataset.cpp
    src/model/fixed_income/vasicek/rate_option.cu
    src/model/fixed_income/vasicek/zero_coupon_bond_option.cu
    src/model/fixed_income/vasicek/dataset.cpp
    src/product/american_option/dataset.cpp
    src/product/athena_autocall/dataset.cpp
    src/product/asian_option/dataset.cpp
    src/product/asset_or_nothing_option/dataset.cpp
    src/product/cliquet/dataset.cpp
    src/product/range_accrual/dataset.cpp
    src/product/digital_option/dataset.cpp
    src/product/double_knock_out_option/dataset.cpp
    src/product/down_and_in_option/dataset.cpp
    src/product/down_and_out_option/dataset.cpp
    src/product/european_option/dataset.cpp
    src/product/rate_option/dataset.cpp
    src/product/gap_option/dataset.cpp
    src/product/forward_start_option/dataset.cpp
    src/product/geometric_asian_option/dataset.cpp
    src/product/lookback_option/dataset.cpp
    src/product/phoenix_autocall/dataset.cpp
    src/product/phoenix_memory_autocall/dataset.cpp
    src/product/straddle/dataset.cpp
    src/product/up_and_in_option/dataset.cpp
    src/product/up_no_touch/dataset.cpp
    src/product/up_one_touch/dataset.cpp
    src/product/up_and_out_option/dataset.cpp
    src/product/zero_coupon_bond_option/dataset.cpp
)
target_include_directories(cuda_workbench PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(
    cuda_workbench PUBLIC nlohmann_json::nlohmann_json ai_factory_dataset_tools
)
target_compile_features(cuda_workbench PUBLIC cxx_std_17)
set_target_properties(cuda_workbench PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
target_compile_options(cuda_workbench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)
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
    catalog/price/equity/heston/european_calls/heston_01__european_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_european_puts_01
    catalog/price/equity/heston/european_puts/heston_01__european_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asian_puts_01
    catalog/price/equity/heston/asian_puts/heston_01__asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_digital_calls_01
    catalog/price/equity/heston/digital_calls/heston_01__digital_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_digital_puts_01
    catalog/price/equity/heston/digital_puts/heston_01__digital_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asset_or_nothing_calls_01
    catalog/price/equity/heston/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asset_or_nothing_puts_01
    catalog/price/equity/heston/asset_or_nothing_puts/heston_01__asset_or_nothing_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_gap_calls_01
    catalog/price/equity/heston/gap_calls/heston_01__gap_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_gap_puts_01
    catalog/price/equity/heston/gap_puts/heston_01__gap_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_straddles_01
    catalog/price/equity/heston/straddles/heston_01__straddles_01__01/generator.cpp
)
add_price_generator(
    generate_g2_caplets_01
    catalog/price/fixed_income/g2/caplets/g2_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_floorlets_01
    catalog/price/fixed_income/g2/floorlets/g2_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_zero_coupon_bond_calls_01
    catalog/price/fixed_income/g2/zero_coupon_bond_calls/g2_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_g2_zero_coupon_bond_puts_01
    catalog/price/fixed_income/g2/zero_coupon_bond_puts/g2_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_caplets_01
    catalog/price/fixed_income/g2_plus_plus/nelson_siegel/caplets/g2_plus_plus_01__nelson_siegel_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_floorlets_01
    catalog/price/fixed_income/g2_plus_plus/nelson_siegel/floorlets/g2_plus_plus_01__nelson_siegel_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_zero_coupon_bond_calls_01
    catalog/price/fixed_income/g2_plus_plus/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_zero_coupon_bond_puts_01
    catalog/price/fixed_income/g2_plus_plus/nelson_siegel/zero_coupon_bond_puts/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_caplets_01
    catalog/price/fixed_income/g2_plus_plus/svensson/caplets/g2_plus_plus_01__svensson_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_floorlets_01
    catalog/price/fixed_income/g2_plus_plus/svensson/floorlets/g2_plus_plus_01__svensson_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_zero_coupon_bond_calls_01
    catalog/price/fixed_income/g2_plus_plus/svensson/zero_coupon_bond_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_zero_coupon_bond_puts_01
    catalog/price/fixed_income/g2_plus_plus/svensson/zero_coupon_bond_puts/g2_plus_plus_01__svensson_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asian_calls_01
    catalog/price/equity/heston/asian_calls/heston_01__asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_athena_autocalls_01
    catalog/price/equity/heston/athena_autocalls/heston_01__athena_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_phoenix_autocalls_01
    catalog/price/equity/heston/phoenix_autocalls/heston_01__phoenix_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_phoenix_memory_autocalls_01
    catalog/price/equity/heston/phoenix_memory_autocalls/heston_01__phoenix_memory_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_cliquets_01
    catalog/price/equity/heston/cliquets/heston_01__cliquets_01__01/generator.cpp
)
add_price_generator(
    generate_heston_range_accruals_01
    catalog/price/equity/heston/range_accruals/heston_01__range_accruals_01__01/generator.cpp
)
add_price_generator(
    generate_heston_lookback_options_01
    catalog/price/equity/heston/lookback_options/heston_01__lookback_options_01__01/generator.cpp
)
add_price_generator(
    generate_heston_american_puts_01
    catalog/price/equity/heston/american_puts/heston_01__american_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_geometric_asian_calls_01
    catalog/price/equity/heston/geometric_asian_calls/heston_01__geometric_asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_geometric_asian_puts_01
    catalog/price/equity/heston/geometric_asian_puts/heston_01__geometric_asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_forward_start_calls_01
    catalog/price/equity/heston/forward_start_calls/heston_01__forward_start_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_forward_start_puts_01
    catalog/price/equity/heston/forward_start_puts/heston_01__forward_start_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_and_out_calls_01
    catalog/price/equity/heston/up_and_out_calls/heston_01__up_and_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_down_and_out_puts_01
    catalog/price/equity/heston/down_and_out_puts/heston_01__down_and_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_and_in_calls_01
    catalog/price/equity/heston/up_and_in_calls/heston_01__up_and_in_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_down_and_in_puts_01
    catalog/price/equity/heston/down_and_in_puts/heston_01__down_and_in_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_one_touches_01
    catalog/price/equity/heston/up_one_touches/heston_01__up_one_touches_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_no_touches_01
    catalog/price/equity/heston/up_no_touches/heston_01__up_no_touches_01__01/generator.cpp
)
add_price_generator(
    generate_heston_double_knock_out_calls_01
    catalog/price/equity/heston/double_knock_out_calls/heston_01__double_knock_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_double_knock_out_puts_01
    catalog/price/equity/heston/double_knock_out_puts/heston_01__double_knock_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_american_calls_01
    catalog/price/equity/heston/american_calls/heston_01__american_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_european_calls_01
    catalog/price/equity/bates/european_calls/bates_01__european_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_european_puts_01
    catalog/price/equity/bates/european_puts/bates_01__european_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asian_puts_01
    catalog/price/equity/bates/asian_puts/bates_01__asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_digital_calls_01
    catalog/price/equity/bates/digital_calls/bates_01__digital_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_digital_puts_01
    catalog/price/equity/bates/digital_puts/bates_01__digital_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asset_or_nothing_calls_01
    catalog/price/equity/bates/asset_or_nothing_calls/bates_01__asset_or_nothing_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asset_or_nothing_puts_01
    catalog/price/equity/bates/asset_or_nothing_puts/bates_01__asset_or_nothing_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_gap_calls_01
    catalog/price/equity/bates/gap_calls/bates_01__gap_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_gap_puts_01
    catalog/price/equity/bates/gap_puts/bates_01__gap_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_straddles_01
    catalog/price/equity/bates/straddles/bates_01__straddles_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asian_calls_01
    catalog/price/equity/bates/asian_calls/bates_01__asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_athena_autocalls_01
    catalog/price/equity/bates/athena_autocalls/bates_01__athena_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_phoenix_autocalls_01
    catalog/price/equity/bates/phoenix_autocalls/bates_01__phoenix_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_phoenix_memory_autocalls_01
    catalog/price/equity/bates/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_cliquets_01
    catalog/price/equity/bates/cliquets/bates_01__cliquets_01__01/generator.cpp
)
add_price_generator(
    generate_bates_range_accruals_01
    catalog/price/equity/bates/range_accruals/bates_01__range_accruals_01__01/generator.cpp
)
add_price_generator(
    generate_bates_lookback_options_01
    catalog/price/equity/bates/lookback_options/bates_01__lookback_options_01__01/generator.cpp
)
add_price_generator(
    generate_bates_american_puts_01
    catalog/price/equity/bates/american_puts/bates_01__american_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_geometric_asian_calls_01
    catalog/price/equity/bates/geometric_asian_calls/bates_01__geometric_asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_geometric_asian_puts_01
    catalog/price/equity/bates/geometric_asian_puts/bates_01__geometric_asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_forward_start_calls_01
    catalog/price/equity/bates/forward_start_calls/bates_01__forward_start_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_forward_start_puts_01
    catalog/price/equity/bates/forward_start_puts/bates_01__forward_start_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_and_out_calls_01
    catalog/price/equity/bates/up_and_out_calls/bates_01__up_and_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_down_and_out_puts_01
    catalog/price/equity/bates/down_and_out_puts/bates_01__down_and_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_and_in_calls_01
    catalog/price/equity/bates/up_and_in_calls/bates_01__up_and_in_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_down_and_in_puts_01
    catalog/price/equity/bates/down_and_in_puts/bates_01__down_and_in_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_one_touches_01
    catalog/price/equity/bates/up_one_touches/bates_01__up_one_touches_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_no_touches_01
    catalog/price/equity/bates/up_no_touches/bates_01__up_no_touches_01__01/generator.cpp
)
add_price_generator(
    generate_bates_double_knock_out_calls_01
    catalog/price/equity/bates/double_knock_out_calls/bates_01__double_knock_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_double_knock_out_puts_01
    catalog/price/equity/bates/double_knock_out_puts/bates_01__double_knock_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_american_calls_01
    catalog/price/equity/bates/american_calls/bates_01__american_calls_01__01/generator.cpp
)

# VG and NIG expose the same product-specialized launcher surface as Heston.
set(_levy_equity_price_families
    american_calls american_puts asian_calls asian_puts
    asset_or_nothing_calls asset_or_nothing_puts athena_autocalls cliquets
    digital_calls digital_puts double_knock_out_calls double_knock_out_puts
    down_and_in_puts down_and_out_puts european_calls european_puts
    forward_start_calls forward_start_puts gap_calls gap_puts
    geometric_asian_calls geometric_asian_puts lookback_options
    phoenix_autocalls phoenix_memory_autocalls range_accruals straddles
    up_and_in_calls up_and_out_calls up_no_touches up_one_touches
)
foreach(_levy_model IN ITEMS variance_gamma normal_inverse_gaussian)
    foreach(_price_family IN LISTS _levy_equity_price_families)
        add_price_generator(
            generate_${_levy_model}_${_price_family}_01
            catalog/price/equity/${_levy_model}/${_price_family}/${_levy_model}_01__${_price_family}_01__01/generator.cpp
        )
    endforeach()
endforeach()
add_price_generator(
    generate_hull_white_nelson_siegel_caplets_01
    catalog/price/fixed_income/hull_white/nelson_siegel/caplets/hull_white_01__nelson_siegel_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_floorlets_01
    catalog/price/fixed_income/hull_white/nelson_siegel/floorlets/hull_white_01__nelson_siegel_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_zero_coupon_bond_calls_01
    catalog/price/fixed_income/hull_white/nelson_siegel/zero_coupon_bond_calls/hull_white_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_zero_coupon_bond_puts_01
    catalog/price/fixed_income/hull_white/nelson_siegel/zero_coupon_bond_puts/hull_white_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_caplets_01
    catalog/price/fixed_income/hull_white/svensson/caplets/hull_white_01__svensson_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_floorlets_01
    catalog/price/fixed_income/hull_white/svensson/floorlets/hull_white_01__svensson_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_zero_coupon_bond_calls_01
    catalog/price/fixed_income/hull_white/svensson/zero_coupon_bond_calls/hull_white_01__svensson_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_zero_coupon_bond_puts_01
    catalog/price/fixed_income/hull_white/svensson/zero_coupon_bond_puts/hull_white_01__svensson_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_caplets_01
    catalog/price/fixed_income/ornstein_uhlenbeck/caplets/ornstein_uhlenbeck_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_floorlets_01
    catalog/price/fixed_income/ornstein_uhlenbeck/floorlets/ornstein_uhlenbeck_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_zero_coupon_bond_calls_01
    catalog/price/fixed_income/ornstein_uhlenbeck/zero_coupon_bond_calls/ornstein_uhlenbeck_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_zero_coupon_bond_puts_01
    catalog/price/fixed_income/ornstein_uhlenbeck/zero_coupon_bond_puts/ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_caplets_01
    catalog/price/fixed_income/vasicek/caplets/vasicek_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_floorlets_01
    catalog/price/fixed_income/vasicek/floorlets/vasicek_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_zero_coupon_bond_calls_01
    catalog/price/fixed_income/vasicek/zero_coupon_bond_calls/vasicek_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_zero_coupon_bond_puts_01
    catalog/price/fixed_income/vasicek/zero_coupon_bond_puts/vasicek_01__zero_coupon_bond_puts_01__01/generator.cpp
)

add_parameter_generator(
    generate_heston_01 catalog/model/equity/heston/heston_01/generator.cpp
)
add_parameter_generator(
    generate_bates_01 catalog/model/equity/bates/bates_01/generator.cpp
)
add_parameter_generator(
    generate_variance_gamma_01
    catalog/model/equity/variance_gamma/variance_gamma_01/generator.cpp
)
add_parameter_generator(
    generate_normal_inverse_gaussian_01
    catalog/model/equity/normal_inverse_gaussian/normal_inverse_gaussian_01/generator.cpp
)
add_parameter_generator(
    generate_g2_01 catalog/model/fixed_income/g2/g2_01/generator.cpp
)
add_parameter_generator(
    generate_g2_plus_plus_01
    catalog/model/fixed_income/g2_plus_plus/g2_plus_plus_01/generator.cpp
)
add_parameter_generator(
    generate_hull_white_01 catalog/model/fixed_income/hull_white/hull_white_01/generator.cpp
)
add_parameter_generator(
    generate_ornstein_uhlenbeck_01
    catalog/model/fixed_income/ornstein_uhlenbeck/ornstein_uhlenbeck_01/generator.cpp
)
add_parameter_generator(
    generate_vasicek_01
    catalog/model/fixed_income/vasicek/vasicek_01/generator.cpp
)
add_parameter_generator(
    generate_nelson_siegel_01
    catalog/curve/nelson_siegel/nelson_siegel_01/generator.cpp
)
add_parameter_generator(
    generate_svensson_01
    catalog/curve/svensson/svensson_01/generator.cpp
)
add_parameter_generator(
    generate_straddles_01
    catalog/product/equity/straddles/straddles_01/generator.cpp
)
add_parameter_generator(
    generate_lookback_options_01
    catalog/product/equity/lookback_options/lookback_options_01/generator.cpp
)
add_parameter_generator(
    generate_up_one_touches_01
    catalog/product/equity/up_one_touches/up_one_touches_01/generator.cpp
)
add_parameter_generator(
    generate_up_no_touches_01
    catalog/product/equity/up_no_touches/up_no_touches_01/generator.cpp
)
add_parameter_generator(
    generate_phoenix_autocalls_01
    catalog/product/equity/phoenix_autocalls/phoenix_autocalls_01/generator.cpp
)
add_parameter_generator(
    generate_phoenix_memory_autocalls_01
    catalog/product/equity/phoenix_memory_autocalls/phoenix_memory_autocalls_01/generator.cpp
)
add_parameter_generator(
    generate_athena_autocalls_01
    catalog/product/equity/athena_autocalls/athena_autocalls_01/generator.cpp
)
add_parameter_generator(
    generate_cliquets_01
    catalog/product/equity/cliquets/cliquets_01/generator.cpp
)
add_parameter_generator(
    generate_range_accruals_01
    catalog/product/equity/range_accruals/range_accruals_01/generator.cpp
)
# Call/put orientation is selected by price generators; product grids are
# shared whenever their construction is identical.
add_parameter_generator(
    generate_european_options_01
    catalog/product/equity/european_options/european_options_01/generator.cpp
)
add_parameter_generator(
    generate_asian_options_01
    catalog/product/equity/asian_options/asian_options_01/generator.cpp
)
add_parameter_generator(
    generate_digital_options_01
    catalog/product/equity/digital_options/digital_options_01/generator.cpp
)
add_parameter_generator(
    generate_asset_or_nothing_options_01
    catalog/product/equity/asset_or_nothing_options/asset_or_nothing_options_01/generator.cpp
)
add_parameter_generator(
    generate_gap_call_options_01
    catalog/product/equity/gap_options/gap_call_options_01/generator.cpp
)
add_parameter_generator(
    generate_gap_put_options_01
    catalog/product/equity/gap_options/gap_put_options_01/generator.cpp
)
add_parameter_generator(
    generate_american_options_01
    catalog/product/equity/american_options/american_options_01/generator.cpp
)
add_parameter_generator(
    generate_geometric_asian_options_01
    catalog/product/equity/geometric_asian_options/geometric_asian_options_01/generator.cpp
)
add_parameter_generator(
    generate_forward_start_options_01
    catalog/product/equity/forward_start_options/forward_start_options_01/generator.cpp
)
add_parameter_generator(
    generate_up_and_out_options_01
    catalog/product/equity/up_and_out_options/up_and_out_options_01/generator.cpp
)
add_parameter_generator(
    generate_down_and_out_options_01
    catalog/product/equity/down_and_out_options/down_and_out_options_01/generator.cpp
)
add_parameter_generator(
    generate_up_and_in_options_01
    catalog/product/equity/up_and_in_options/up_and_in_options_01/generator.cpp
)
add_parameter_generator(
    generate_down_and_in_options_01
    catalog/product/equity/down_and_in_options/down_and_in_options_01/generator.cpp
)
add_parameter_generator(
    generate_double_knock_out_options_01
    catalog/product/equity/double_knock_out_options/double_knock_out_options_01/generator.cpp
)
add_parameter_generator(
    generate_rate_options_01
    catalog/product/fixed_income/rate_options/rate_options_01/generator.cpp
)
add_parameter_generator(
    generate_zero_coupon_bond_options_01
    catalog/product/fixed_income/zero_coupon_bond_options/zero_coupon_bond_options_01/generator.cpp
)

if(BUILD_TESTING)
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

    add_executable(test_dataset_loaders tests/dataset_loaders_test.cpp)
    target_link_libraries(test_dataset_loaders PRIVATE cuda_workbench)
    target_compile_features(test_dataset_loaders PRIVATE cxx_std_17)
    add_test(NAME dataset_loaders COMMAND test_dataset_loaders)
    set_tests_properties(dataset_loaders PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;dataset"
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
                function(add_quantlib_validation_test name module dataset label)
                    add_test(
                        NAME quantlib_${name}
                        COMMAND ${Python3_EXECUTABLE} -m ${module} ${dataset}
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
                        datasets/price/fixed_income/ornstein_uhlenbeck/${product_folder}/ornstein_uhlenbeck_01__${product_folder}_01__01.json
                        ornstein_uhlenbeck
                    )
                    add_quantlib_validation_test(
                        vasicek_${product}
                        validation.quantlib.model.fixed_income.vasicek.${product}
                        datasets/price/fixed_income/vasicek/${product_folder}/vasicek_01__${product_folder}_01__01.json
                        vasicek
                    )
                    add_quantlib_validation_test(
                        hull_white_nelson_siegel_${product}
                        validation.quantlib.model.fixed_income.hull_white.nelson_siegel.${product}
                        datasets/price/fixed_income/hull_white/nelson_siegel/${product_folder}/hull_white_01__nelson_siegel_01__${product_folder}_01__01.json
                        hull_white
                    )
                    add_quantlib_validation_test(
                        hull_white_svensson_${product}
                        validation.quantlib.model.fixed_income.hull_white.svensson.${product}
                        datasets/price/fixed_income/hull_white/svensson/${product_folder}/hull_white_01__svensson_01__${product_folder}_01__01.json
                        hull_white
                    )
                    add_quantlib_validation_test(
                        g2_${product}
                        validation.quantlib.model.fixed_income.g2.${product}
                        datasets/price/fixed_income/g2/${product_folder}/g2_01__${product_folder}_01__01.json
                        g2
                    )
                    add_quantlib_validation_test(
                        g2_plus_plus_nelson_siegel_${product}
                        validation.quantlib.model.fixed_income.g2_plus_plus.nelson_siegel.${product}
                        datasets/price/fixed_income/g2_plus_plus/nelson_siegel/${product_folder}/g2_plus_plus_01__nelson_siegel_01__${product_folder}_01__01.json
                        g2_plus_plus
                    )
                    add_quantlib_validation_test(
                        g2_plus_plus_svensson_${product}
                        validation.quantlib.model.fixed_income.g2_plus_plus.svensson.${product}
                        datasets/price/fixed_income/g2_plus_plus/svensson/${product_folder}/g2_plus_plus_01__svensson_01__${product_folder}_01__01.json
                        g2_plus_plus
                    )
                endforeach()

                add_quantlib_validation_test(
                    heston_european_call
                    validation.quantlib.model.equity.heston.european_call
                    datasets/price/equity/heston/european_calls/heston_01__european_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_european_put
                    validation.quantlib.model.equity.heston.european_put
                    datasets/price/equity/heston/european_puts/heston_01__european_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_straddle
                    validation.quantlib.model.equity.heston.straddle
                    datasets/price/equity/heston/straddles/heston_01__straddles_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_digital_call
                    validation.quantlib.model.equity.heston.digital_call
                    datasets/price/equity/heston/digital_calls/heston_01__digital_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_digital_put
                    validation.quantlib.model.equity.heston.digital_put
                    datasets/price/equity/heston/digital_puts/heston_01__digital_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_asset_or_nothing_call
                    validation.quantlib.model.equity.heston.asset_or_nothing_call
                    datasets/price/equity/heston/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_asset_or_nothing_put
                    validation.quantlib.model.equity.heston.asset_or_nothing_put
                    datasets/price/equity/heston/asset_or_nothing_puts/heston_01__asset_or_nothing_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_gap_call
                    validation.quantlib.model.equity.heston.gap_call
                    datasets/price/equity/heston/gap_calls/heston_01__gap_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_gap_put
                    validation.quantlib.model.equity.heston.gap_put
                    datasets/price/equity/heston/gap_puts/heston_01__gap_puts_01__01.json
                    heston
                )

                function(add_bates_validation_test product folder)
                    add_quantlib_validation_test(
                        bates_${product}
                        validation.quantlib.model.equity.bates.${product}
                        datasets/price/equity/bates/${folder}/bates_01__${folder}_01__01.json
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
                            datasets/price/equity/variance_gamma/${folder}/variance_gamma_01__${folder}_01__01.json
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
                        datasets/price/equity/heston/geometric_asian_calls/heston_01__geometric_asian_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_geometric_asian_put
                        validation.quantlib.model.equity.heston.geometric_asian_put
                        datasets/price/equity/heston/geometric_asian_puts/heston_01__geometric_asian_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_asian_call
                        validation.quantlib.model.equity.heston.asian_call
                        datasets/price/equity/heston/asian_calls/heston_01__asian_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_american_put
                        validation.quantlib.model.equity.heston.american_put
                        datasets/price/equity/heston/american_puts/heston_01__american_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_asian_put
                        validation.quantlib.model.equity.heston.asian_put
                        datasets/price/equity/heston/asian_puts/heston_01__asian_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_and_out_call
                        validation.quantlib.model.equity.heston.up_and_out_call
                        datasets/price/equity/heston/up_and_out_calls/heston_01__up_and_out_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_down_and_out_put
                        validation.quantlib.model.equity.heston.down_and_out_put
                        datasets/price/equity/heston/down_and_out_puts/heston_01__down_and_out_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_and_in_call
                        validation.quantlib.model.equity.heston.up_and_in_call
                        datasets/price/equity/heston/up_and_in_calls/heston_01__up_and_in_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_down_and_in_put
                        validation.quantlib.model.equity.heston.down_and_in_put
                        datasets/price/equity/heston/down_and_in_puts/heston_01__down_and_in_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_one_touch
                        validation.quantlib.model.equity.heston.up_one_touch
                        datasets/price/equity/heston/up_one_touches/heston_01__up_one_touches_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_no_touch
                        validation.quantlib.model.equity.heston.up_no_touch
                        datasets/price/equity/heston/up_no_touches/heston_01__up_no_touches_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_double_knock_out_call
                        validation.quantlib.model.equity.heston.double_knock_out_call
                        datasets/price/equity/heston/double_knock_out_calls/heston_01__double_knock_out_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_double_knock_out_put
                        validation.quantlib.model.equity.heston.double_knock_out_put
                        datasets/price/equity/heston/double_knock_out_puts/heston_01__double_knock_out_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_american_call
                        validation.quantlib.model.equity.heston.american_call
                        datasets/price/equity/heston/american_calls/heston_01__american_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_forward_start_call
                        validation.quantlib.model.equity.heston.forward_start_call
                        datasets/price/equity/heston/forward_start_calls/heston_01__forward_start_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_forward_start_put
                        validation.quantlib.model.equity.heston.forward_start_put
                        datasets/price/equity/heston/forward_start_puts/heston_01__forward_start_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_athena_autocall
                        validation.quantlib.model.equity.heston.athena_autocall
                        datasets/price/equity/heston/athena_autocalls/heston_01__athena_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_phoenix_autocall
                        validation.quantlib.model.equity.heston.phoenix_autocall
                        datasets/price/equity/heston/phoenix_autocalls/heston_01__phoenix_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_phoenix_memory_autocall
                        validation.quantlib.model.equity.heston.phoenix_memory_autocall
                        datasets/price/equity/heston/phoenix_memory_autocalls/heston_01__phoenix_memory_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_cliquet
                        validation.quantlib.model.equity.heston.cliquet
                        datasets/price/equity/heston/cliquets/heston_01__cliquets_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_range_accrual
                        validation.quantlib.model.equity.heston.range_accrual
                        datasets/price/equity/heston/range_accruals/heston_01__range_accruals_01__01.json
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
                    add_bates_validation_test(american_call american_calls)
                    add_bates_validation_test(american_put american_puts)
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
        philox_cuda tests/philox_cuda_test.cu philox 30
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
        levy_dynamics_cuda
        tests/levy_dynamics_cuda_test.cu
        levy_dynamics
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
        cuda_kernel_diagnostics
        tests/cuda_kernel_diagnostics_test.cu
        kernel_diagnostics
        30
    )

endif()
