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
    "Root of NVIDIA mathDx (enables the cuFFTDx rough-Bergomi pricer)"
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

# Host-only JSON validation and serialization used by every dataset family.
add_library(ai_factory_dataset_core STATIC
    tools/datasets/dataset.cpp
    tools/datasets/dataset_validation.cpp
)
target_include_directories(ai_factory_dataset_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    ai_factory_dataset_core PUBLIC nlohmann_json::nlohmann_json
)
target_compile_features(ai_factory_dataset_core PUBLIC cxx_std_23)
target_compile_options(ai_factory_dataset_core PRIVATE -O3)

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
    target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${target} PUBLIC ai_factory_dataset_core)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_compile_options(${target} PRIVATE -O3)
    list(APPEND _ai_factory_generation_targets ${target})
endforeach()

# Compatibility umbrella for parameter generators and external consumers.
add_library(ai_factory_dataset_tools INTERFACE)
target_link_libraries(ai_factory_dataset_tools INTERFACE
    ai_factory_dataset_core
    ${_ai_factory_generation_targets}
)

include(cmake/AIFactoryTargets.cmake)

# Register one host-only parameter generator with shared build settings.
add_custom_target(parameter_generators)
add_custom_target(price_generators)

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
    list(REMOVE_DUPLICATES dependencies)
    set(${output} ${dependencies} PARENT_SCOPE)
endfunction()

function(add_parameter_generator target source)
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    ai_factory_collect_generation_dependencies(dependencies ${source})
    target_link_libraries(
        ${target} PRIVATE ai_factory_dataset_core ${dependencies}
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    add_dependencies(parameter_generators ${target})
endfunction()

# Register one CUDA price generator with shared build settings.
function(add_price_generator target source)
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    ai_factory_collect_source_dependencies(dependencies ${source})
    if(NOT dependencies)
        message(FATAL_ERROR
            "No compiled pricing unit found for generator ${target}: ${source}"
        )
    endif()
    target_link_libraries(
        ${target} PRIVATE ${dependencies} ai_factory_dataset_core
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    add_dependencies(price_generators ${target})
endfunction()

add_price_generator(
    generate_heston_european_calls_01
    catalog/model/equity/heston/prices/european_calls/heston_01__european_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_european_puts_01
    catalog/model/equity/heston/prices/european_puts/heston_01__european_puts_01__01/generator.cpp
)
add_price_generator(
    generate_rough_bergomi_european_calls_01
    catalog/model/equity/rough_bergomi/prices/european_calls/rough_bergomi_01__european_calls_01__01/generator.cpp
)
add_price_generator(
    generate_rough_bergomi_european_puts_01
    catalog/model/equity/rough_bergomi/prices/european_puts/rough_bergomi_01__european_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asian_puts_01
    catalog/model/equity/heston/prices/asian_puts/heston_01__asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_digital_calls_01
    catalog/model/equity/heston/prices/digital_calls/heston_01__digital_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_digital_puts_01
    catalog/model/equity/heston/prices/digital_puts/heston_01__digital_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asset_or_nothing_calls_01
    catalog/model/equity/heston/prices/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asset_or_nothing_puts_01
    catalog/model/equity/heston/prices/asset_or_nothing_puts/heston_01__asset_or_nothing_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_gap_calls_01
    catalog/model/equity/heston/prices/gap_calls/heston_01__gap_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_gap_puts_01
    catalog/model/equity/heston/prices/gap_puts/heston_01__gap_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_straddles_01
    catalog/model/equity/heston/prices/straddles/heston_01__straddles_01__01/generator.cpp
)
add_price_generator(
    generate_g2_caplets_01
    catalog/model/fixed_income/g2/prices/caplets/g2_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_floorlets_01
    catalog/model/fixed_income/g2/prices/floorlets/g2_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_zero_coupon_bond_calls_01
    catalog/model/fixed_income/g2/prices/zero_coupon_bond_calls/g2_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_g2_zero_coupon_bond_puts_01
    catalog/model/fixed_income/g2/prices/zero_coupon_bond_puts/g2_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_caplets_01
    catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/caplets/g2_plus_plus_01__nelson_siegel_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_floorlets_01
    catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/floorlets/g2_plus_plus_01__nelson_siegel_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_zero_coupon_bond_calls_01
    catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_nelson_siegel_zero_coupon_bond_puts_01
    catalog/model/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_puts/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_caplets_01
    catalog/model/fixed_income/g2_plus_plus/prices/svensson/caplets/g2_plus_plus_01__svensson_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_floorlets_01
    catalog/model/fixed_income/g2_plus_plus/prices/svensson/floorlets/g2_plus_plus_01__svensson_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_zero_coupon_bond_calls_01
    catalog/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_calls/g2_plus_plus_01__svensson_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_g2_plus_plus_svensson_zero_coupon_bond_puts_01
    catalog/model/fixed_income/g2_plus_plus/prices/svensson/zero_coupon_bond_puts/g2_plus_plus_01__svensson_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_asian_calls_01
    catalog/model/equity/heston/prices/asian_calls/heston_01__asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_athena_autocalls_01
    catalog/model/equity/heston/prices/athena_autocalls/heston_01__athena_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_phoenix_autocalls_01
    catalog/model/equity/heston/prices/phoenix_autocalls/heston_01__phoenix_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_phoenix_memory_autocalls_01
    catalog/model/equity/heston/prices/phoenix_memory_autocalls/heston_01__phoenix_memory_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_cliquets_01
    catalog/model/equity/heston/prices/cliquets/heston_01__cliquets_01__01/generator.cpp
)
add_price_generator(
    generate_heston_range_accruals_01
    catalog/model/equity/heston/prices/range_accruals/heston_01__range_accruals_01__01/generator.cpp
)
add_price_generator(
    generate_heston_lookback_options_01
    catalog/model/equity/heston/prices/lookback_options/heston_01__lookback_options_01__01/generator.cpp
)
add_price_generator(
    generate_heston_american_puts_01
    catalog/model/equity/heston/prices/american_puts/heston_01__american_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_geometric_asian_calls_01
    catalog/model/equity/heston/prices/geometric_asian_calls/heston_01__geometric_asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_geometric_asian_puts_01
    catalog/model/equity/heston/prices/geometric_asian_puts/heston_01__geometric_asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_forward_start_calls_01
    catalog/model/equity/heston/prices/forward_start_calls/heston_01__forward_start_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_forward_start_puts_01
    catalog/model/equity/heston/prices/forward_start_puts/heston_01__forward_start_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_and_out_calls_01
    catalog/model/equity/heston/prices/up_and_out_calls/heston_01__up_and_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_down_and_out_puts_01
    catalog/model/equity/heston/prices/down_and_out_puts/heston_01__down_and_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_and_in_calls_01
    catalog/model/equity/heston/prices/up_and_in_calls/heston_01__up_and_in_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_down_and_in_puts_01
    catalog/model/equity/heston/prices/down_and_in_puts/heston_01__down_and_in_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_one_touches_01
    catalog/model/equity/heston/prices/up_one_touches/heston_01__up_one_touches_01__01/generator.cpp
)
add_price_generator(
    generate_heston_up_no_touches_01
    catalog/model/equity/heston/prices/up_no_touches/heston_01__up_no_touches_01__01/generator.cpp
)
add_price_generator(
    generate_heston_double_knock_out_calls_01
    catalog/model/equity/heston/prices/double_knock_out_calls/heston_01__double_knock_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_heston_double_knock_out_puts_01
    catalog/model/equity/heston/prices/double_knock_out_puts/heston_01__double_knock_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_heston_american_calls_01
    catalog/model/equity/heston/prices/american_calls/heston_01__american_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_european_calls_01
    catalog/model/equity/bates/prices/european_calls/bates_01__european_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_european_puts_01
    catalog/model/equity/bates/prices/european_puts/bates_01__european_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asian_puts_01
    catalog/model/equity/bates/prices/asian_puts/bates_01__asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_digital_calls_01
    catalog/model/equity/bates/prices/digital_calls/bates_01__digital_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_digital_puts_01
    catalog/model/equity/bates/prices/digital_puts/bates_01__digital_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asset_or_nothing_calls_01
    catalog/model/equity/bates/prices/asset_or_nothing_calls/bates_01__asset_or_nothing_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asset_or_nothing_puts_01
    catalog/model/equity/bates/prices/asset_or_nothing_puts/bates_01__asset_or_nothing_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_gap_calls_01
    catalog/model/equity/bates/prices/gap_calls/bates_01__gap_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_gap_puts_01
    catalog/model/equity/bates/prices/gap_puts/bates_01__gap_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_straddles_01
    catalog/model/equity/bates/prices/straddles/bates_01__straddles_01__01/generator.cpp
)
add_price_generator(
    generate_bates_asian_calls_01
    catalog/model/equity/bates/prices/asian_calls/bates_01__asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_athena_autocalls_01
    catalog/model/equity/bates/prices/athena_autocalls/bates_01__athena_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_phoenix_autocalls_01
    catalog/model/equity/bates/prices/phoenix_autocalls/bates_01__phoenix_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_phoenix_memory_autocalls_01
    catalog/model/equity/bates/prices/phoenix_memory_autocalls/bates_01__phoenix_memory_autocalls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_cliquets_01
    catalog/model/equity/bates/prices/cliquets/bates_01__cliquets_01__01/generator.cpp
)
add_price_generator(
    generate_bates_range_accruals_01
    catalog/model/equity/bates/prices/range_accruals/bates_01__range_accruals_01__01/generator.cpp
)
add_price_generator(
    generate_bates_lookback_options_01
    catalog/model/equity/bates/prices/lookback_options/bates_01__lookback_options_01__01/generator.cpp
)
add_price_generator(
    generate_bates_american_puts_01
    catalog/model/equity/bates/prices/american_puts/bates_01__american_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_geometric_asian_calls_01
    catalog/model/equity/bates/prices/geometric_asian_calls/bates_01__geometric_asian_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_geometric_asian_puts_01
    catalog/model/equity/bates/prices/geometric_asian_puts/bates_01__geometric_asian_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_forward_start_calls_01
    catalog/model/equity/bates/prices/forward_start_calls/bates_01__forward_start_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_forward_start_puts_01
    catalog/model/equity/bates/prices/forward_start_puts/bates_01__forward_start_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_and_out_calls_01
    catalog/model/equity/bates/prices/up_and_out_calls/bates_01__up_and_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_down_and_out_puts_01
    catalog/model/equity/bates/prices/down_and_out_puts/bates_01__down_and_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_and_in_calls_01
    catalog/model/equity/bates/prices/up_and_in_calls/bates_01__up_and_in_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_down_and_in_puts_01
    catalog/model/equity/bates/prices/down_and_in_puts/bates_01__down_and_in_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_one_touches_01
    catalog/model/equity/bates/prices/up_one_touches/bates_01__up_one_touches_01__01/generator.cpp
)
add_price_generator(
    generate_bates_up_no_touches_01
    catalog/model/equity/bates/prices/up_no_touches/bates_01__up_no_touches_01__01/generator.cpp
)
add_price_generator(
    generate_bates_double_knock_out_calls_01
    catalog/model/equity/bates/prices/double_knock_out_calls/bates_01__double_knock_out_calls_01__01/generator.cpp
)
add_price_generator(
    generate_bates_double_knock_out_puts_01
    catalog/model/equity/bates/prices/double_knock_out_puts/bates_01__double_knock_out_puts_01__01/generator.cpp
)
add_price_generator(
    generate_bates_american_calls_01
    catalog/model/equity/bates/prices/american_calls/bates_01__american_calls_01__01/generator.cpp
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
            catalog/model/equity/${_levy_model}/prices/${_price_family}/${_levy_model}_01__${_price_family}_01__01/generator.cpp
        )
    endforeach()
endforeach()

# Black-Scholes path-dependent products retain the one-block-per-price Monte
# Carlo surface; deterministic products are registered separately below.
set(_black_scholes_monte_carlo_price_families
    asian_calls asian_puts athena_autocalls cliquets
    double_knock_out_calls double_knock_out_puts
    down_and_in_puts down_and_out_puts lookback_options
    phoenix_autocalls phoenix_memory_autocalls
    up_and_in_calls up_and_out_calls up_no_touches up_one_touches
)
foreach(_price_family IN LISTS _black_scholes_monte_carlo_price_families)
    add_price_generator(
        generate_black_scholes_${_price_family}_01
        catalog/model/equity/black_scholes/prices/${_price_family}/black_scholes_01__${_price_family}_01__01/generator.cpp
    )
endforeach()
set(_black_scholes_analytical_price_families
    asset_or_nothing_calls asset_or_nothing_puts
    digital_calls digital_puts european_calls european_puts
    forward_start_calls forward_start_puts gap_calls gap_puts
    geometric_asian_calls geometric_asian_puts range_accruals straddles
)
foreach(_price_family IN LISTS _black_scholes_analytical_price_families)
    add_price_generator(
        generate_black_scholes_${_price_family}_01
        catalog/model/equity/black_scholes/prices/${_price_family}/black_scholes_01__${_price_family}_01__01/generator.cpp
    )
endforeach()

# The V1 equity surface is Monte Carlo only for these four models. Calls and
# puts remain separate datasets while sharing their option-side CUDA template.
set(_new_equity_monte_carlo_price_families
    asian_calls asian_puts asset_or_nothing_calls asset_or_nothing_puts
    athena_autocalls cliquets digital_calls digital_puts
    double_knock_out_calls double_knock_out_puts down_and_in_puts
    down_and_out_puts european_calls european_puts forward_start_calls
    forward_start_puts gap_calls gap_puts geometric_asian_calls
    geometric_asian_puts lookback_options phoenix_autocalls
    phoenix_memory_autocalls range_accruals straddles up_and_in_calls
    up_and_out_calls up_no_touches up_one_touches
)
foreach(_equity_model IN ITEMS merton kou cev schobel_zhu)
    foreach(_price_family IN LISTS _new_equity_monte_carlo_price_families)
        add_price_generator(
            generate_${_equity_model}_${_price_family}_01
            catalog/model/equity/${_equity_model}/prices/${_price_family}/${_equity_model}_01__${_price_family}_01__01/generator.cpp
        )
    endforeach()
endforeach()
add_price_generator(
    generate_hull_white_nelson_siegel_caplets_01
    catalog/model/fixed_income/hull_white/prices/nelson_siegel/caplets/hull_white_01__nelson_siegel_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_floorlets_01
    catalog/model/fixed_income/hull_white/prices/nelson_siegel/floorlets/hull_white_01__nelson_siegel_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_zero_coupon_bond_calls_01
    catalog/model/fixed_income/hull_white/prices/nelson_siegel/zero_coupon_bond_calls/hull_white_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_zero_coupon_bond_puts_01
    catalog/model/fixed_income/hull_white/prices/nelson_siegel/zero_coupon_bond_puts/hull_white_01__nelson_siegel_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_caplets_01
    catalog/model/fixed_income/hull_white/prices/svensson/caplets/hull_white_01__svensson_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_floorlets_01
    catalog/model/fixed_income/hull_white/prices/svensson/floorlets/hull_white_01__svensson_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_zero_coupon_bond_calls_01
    catalog/model/fixed_income/hull_white/prices/svensson/zero_coupon_bond_calls/hull_white_01__svensson_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_zero_coupon_bond_puts_01
    catalog/model/fixed_income/hull_white/prices/svensson/zero_coupon_bond_puts/hull_white_01__svensson_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_cir_caplets_01
    catalog/model/fixed_income/cir/prices/caplets/cir_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_cir_floorlets_01
    catalog/model/fixed_income/cir/prices/floorlets/cir_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_cir_zero_coupon_bond_calls_01
    catalog/model/fixed_income/cir/prices/zero_coupon_bond_calls/cir_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_cir_zero_coupon_bond_puts_01
    catalog/model/fixed_income/cir/prices/zero_coupon_bond_puts/cir_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_caplets_01
    catalog/model/fixed_income/ornstein_uhlenbeck/prices/caplets/ornstein_uhlenbeck_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_floorlets_01
    catalog/model/fixed_income/ornstein_uhlenbeck/prices/floorlets/ornstein_uhlenbeck_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_zero_coupon_bond_calls_01
    catalog/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_calls/ornstein_uhlenbeck_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_zero_coupon_bond_puts_01
    catalog/model/fixed_income/ornstein_uhlenbeck/prices/zero_coupon_bond_puts/ornstein_uhlenbeck_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_caplets_01
    catalog/model/fixed_income/vasicek/prices/caplets/vasicek_01__caplets_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_floorlets_01
    catalog/model/fixed_income/vasicek/prices/floorlets/vasicek_01__floorlets_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_zero_coupon_bond_calls_01
    catalog/model/fixed_income/vasicek/prices/zero_coupon_bond_calls/vasicek_01__zero_coupon_bond_calls_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_zero_coupon_bond_puts_01
    catalog/model/fixed_income/vasicek/prices/zero_coupon_bond_puts/vasicek_01__zero_coupon_bond_puts_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_european_payer_swaptions_01
    catalog/model/fixed_income/ornstein_uhlenbeck/prices/european_payer_swaptions/ornstein_uhlenbeck_01__european_payer_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_ornstein_uhlenbeck_european_receiver_swaptions_01
    catalog/model/fixed_income/ornstein_uhlenbeck/prices/european_receiver_swaptions/ornstein_uhlenbeck_01__european_receiver_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_european_payer_swaptions_01
    catalog/model/fixed_income/vasicek/prices/european_payer_swaptions/vasicek_01__european_payer_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_vasicek_european_receiver_swaptions_01
    catalog/model/fixed_income/vasicek/prices/european_receiver_swaptions/vasicek_01__european_receiver_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_cir_european_payer_swaptions_01
    catalog/model/fixed_income/cir/prices/european_payer_swaptions/cir_01__european_payer_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_cir_european_receiver_swaptions_01
    catalog/model/fixed_income/cir/prices/european_receiver_swaptions/cir_01__european_receiver_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_european_payer_swaptions_01
    catalog/model/fixed_income/hull_white/prices/nelson_siegel/european_payer_swaptions/hull_white_01__nelson_siegel_01__european_payer_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_nelson_siegel_european_receiver_swaptions_01
    catalog/model/fixed_income/hull_white/prices/nelson_siegel/european_receiver_swaptions/hull_white_01__nelson_siegel_01__european_receiver_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_european_payer_swaptions_01
    catalog/model/fixed_income/hull_white/prices/svensson/european_payer_swaptions/hull_white_01__svensson_01__european_payer_swaptions_01__01/generator.cpp
)
add_price_generator(
    generate_hull_white_svensson_european_receiver_swaptions_01
    catalog/model/fixed_income/hull_white/prices/svensson/european_receiver_swaptions/hull_white_01__svensson_01__european_receiver_swaptions_01__01/generator.cpp
)

add_parameter_generator(
    generate_heston_01 catalog/model/equity/heston/parameters/heston_01/generator.cpp
)
add_parameter_generator(
    generate_black_scholes_01
    catalog/model/equity/black_scholes/parameters/black_scholes_01/generator.cpp
)
add_parameter_generator(
    generate_bates_01 catalog/model/equity/bates/parameters/bates_01/generator.cpp
)
add_parameter_generator(
    generate_variance_gamma_01
    catalog/model/equity/variance_gamma/parameters/variance_gamma_01/generator.cpp
)
add_parameter_generator(
    generate_normal_inverse_gaussian_01
    catalog/model/equity/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01/generator.cpp
)
add_parameter_generator(
    generate_merton_01 catalog/model/equity/merton/parameters/merton_01/generator.cpp
)
add_parameter_generator(
    generate_kou_01 catalog/model/equity/kou/parameters/kou_01/generator.cpp
)
add_parameter_generator(
    generate_cev_01 catalog/model/equity/cev/parameters/cev_01/generator.cpp
)
add_parameter_generator(
    generate_schobel_zhu_01
    catalog/model/equity/schobel_zhu/parameters/schobel_zhu_01/generator.cpp
)
add_parameter_generator(
    generate_rough_bergomi_01
    catalog/model/equity/rough_bergomi/parameters/rough_bergomi_01/generator.cpp
)
add_parameter_generator(
    generate_g2_01 catalog/model/fixed_income/g2/parameters/g2_01/generator.cpp
)
add_parameter_generator(
    generate_g2_plus_plus_01
    catalog/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01/generator.cpp
)
add_parameter_generator(
    generate_hull_white_01 catalog/model/fixed_income/hull_white/parameters/hull_white_01/generator.cpp
)
add_parameter_generator(
    generate_ornstein_uhlenbeck_01
    catalog/model/fixed_income/ornstein_uhlenbeck/parameters/ornstein_uhlenbeck_01/generator.cpp
)
add_parameter_generator(
    generate_vasicek_01
    catalog/model/fixed_income/vasicek/parameters/vasicek_01/generator.cpp
)
add_parameter_generator(
    generate_cir_01
    catalog/model/fixed_income/cir/parameters/cir_01/generator.cpp
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
add_parameter_generator(
    generate_european_swaptions_01
    catalog/product/fixed_income/european_swaptions/european_swaptions_01/generator.cpp
)

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
        noncentral_chi_square_cuda
        tests/noncentral_chi_square_cuda_test.cu
        distributions
        30
    )
    add_cuda_workbench_test(
        cir_cuda tests/cir_cuda_test.cu cir 30
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
        levy_dynamics_cuda
        tests/levy_dynamics_cuda_test.cu
        levy_dynamics
        30
    )
    add_cuda_workbench_test(
        new_equity_dynamics_cuda
        tests/new_equity_dynamics_cuda_test.cu
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
        rough_bergomi_european_option_cuda
        tests/rough_bergomi_european_option_cuda_test.cpp
        rough_bergomi_products
        60
    )
    if(AI_FACTORY_MATHDX_ROOT)
        add_cuda_workbench_test(
            rough_bergomi_european_option_fft_cuda
            tests/rough_bergomi_european_option_fft_cuda_test.cpp
            rough_bergomi_products
            120
        )
    endif()
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
        cuda_kernel_diagnostics
        tests/cuda_kernel_diagnostics_test.cu
        kernel_diagnostics
        30
    )

endif()
