# Fine-grained build graph for CUDA pricing units and JSON dataset loaders.
#
# Numerical implementation files such as dynamics.cu and analytics.cu remain
# force-inlined includes of their consuming kernels. Only public launch units
# are compiled here, so no relocatable-device-code boundary is introduced.

function(ai_factory_configure_host_library target)
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(${target} PUBLIC nlohmann_json::nlohmann_json)
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endfunction()

function(ai_factory_configure_cuda_library target)
    ai_factory_configure_host_library(${target})
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED YES
    )
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
endfunction()

add_library(ai_factory_runtime STATIC EXCLUDE_FROM_ALL
    src/common/cuda_kernel_diagnostics.cpp
)
ai_factory_configure_host_library(ai_factory_runtime)

add_library(ai_factory_longstaff_schwartz STATIC EXCLUDE_FROM_ALL
    src/common/longstaff_schwartz/exercise_schedule.cu
    src/common/longstaff_schwartz/launch.cu
    src/common/longstaff_schwartz/workspace.cu
)
ai_factory_configure_cuda_library(ai_factory_longstaff_schwartz)
target_link_libraries(
    ai_factory_longstaff_schwartz PUBLIC ai_factory_runtime
)

function(ai_factory_add_dataset_library target source)
    add_library(${target} STATIC EXCLUDE_FROM_ALL ${source})
    ai_factory_configure_host_library(${target})
    target_link_libraries(${target} PUBLIC ai_factory_dataset_core)
    set_property(
        GLOBAL APPEND PROPERTY AI_FACTORY_DATASET_TARGETS ${target}
    )
endfunction()

set(_ai_factory_curves nelson_siegel svensson)
foreach(curve IN LISTS _ai_factory_curves)
    ai_factory_add_dataset_library(
        ai_factory_curve_${curve}_dataset
        src/curve/${curve}/dataset.cpp
    )
endforeach()

set(_ai_factory_products
    american_option
    asian_option
    asset_or_nothing_option
    athena_autocall
    cliquet
    digital_option
    double_knock_out_option
    down_and_in_option
    down_and_out_option
    european_option
    european_swaption
    forward_start_option
    gap_option
    geometric_asian_option
    lookback_option
    phoenix_autocall
    phoenix_memory_autocall
    range_accrual
    rate_option
    straddle
    up_and_in_option
    up_and_out_option
    up_no_touch
    up_one_touch
    zero_coupon_bond_option
)
foreach(product IN LISTS _ai_factory_products)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/product/${product}/dataset.cpp")
        ai_factory_add_dataset_library(
            ai_factory_product_${product}_dataset
            src/product/${product}/dataset.cpp
        )
    endif()
endforeach()

set(_ai_factory_equity_models
    bates
    black_scholes
    cev
    heston
    kou
    merton
    normal_inverse_gaussian
    rough_bergomi
    schobel_zhu
    variance_gamma
)
set(_ai_factory_fixed_income_models
    cir
    g2
    g2_plus_plus
    hull_white
    ornstein_uhlenbeck
    vasicek
)

foreach(model IN LISTS _ai_factory_equity_models)
    ai_factory_add_dataset_library(
        ai_factory_equity_${model}_dataset
        src/model/equity/${model}/dataset.cpp
    )
endforeach()
foreach(model IN LISTS _ai_factory_fixed_income_models)
    ai_factory_add_dataset_library(
        ai_factory_fixed_income_${model}_dataset
        src/model/fixed_income/${model}/dataset.cpp
    )
endforeach()

# Add one independently compilable public CUDA launcher. The unit links only
# the loaders that its generators need; changing another model or product does
# not enter its dependency graph.
function(ai_factory_add_cuda_unit domain unit_path)
    string(REPLACE "/" "_" unit_id "${unit_path}")
    set(target ai_factory_${domain}_${unit_id})
    set(source src/model/${domain}/${unit_path}.cu)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        return()
    endif()

    add_library(${target} STATIC EXCLUDE_FROM_ALL ${source})
    ai_factory_configure_cuda_library(${target})

    string(REGEX REPLACE "^([^/]+).*$" "\\1" model "${unit_path}")
    get_filename_component(product "${unit_path}" NAME)
    set(dependencies
        ai_factory_runtime
        ai_factory_${domain}_${model}_dataset
    )
    if(TARGET ai_factory_product_${product}_dataset)
        list(APPEND dependencies ai_factory_product_${product}_dataset)
    elseif(product STREQUAL "european_option_fft")
        list(APPEND dependencies ai_factory_product_european_option_dataset)
    endif()
    if(unit_path MATCHES "^[^/]+/(nelson_siegel|svensson)/")
        list(APPEND dependencies
            ai_factory_curve_${CMAKE_MATCH_1}_dataset
        )
    endif()
    if(product STREQUAL "american_option")
        list(APPEND dependencies ai_factory_longstaff_schwartz)
    endif()
    target_link_libraries(${target} PUBLIC ${dependencies})

    set_target_properties(${target} PROPERTIES
        AI_FACTORY_DOMAIN "${domain}"
        AI_FACTORY_MODEL "${model}"
        AI_FACTORY_UNIT "${product}"
    )
    set_property(
        GLOBAL APPEND PROPERTY AI_FACTORY_CUDA_UNIT_TARGETS ${target}
    )
    set_property(
        GLOBAL APPEND
        PROPERTY AI_FACTORY_${domain}_${model}_CUDA_UNITS ${target}
    )
endfunction()

set(_ai_factory_equity_products
    american_option
    asian_option
    asset_or_nothing_option
    athena_autocall
    cliquet
    digital_option
    double_knock_out_option
    down_and_in_option
    down_and_out_option
    european_option
    forward_start_option
    gap_option
    geometric_asian_option
    lookback_option
    phoenix_autocall
    phoenix_memory_autocall
    range_accrual
    straddle
    up_and_in_option
    up_and_out_option
    up_no_touch
    up_one_touch
)
foreach(model IN LISTS _ai_factory_equity_models)
    ai_factory_add_cuda_unit(equity ${model}/sample)
    foreach(product IN LISTS _ai_factory_equity_products)
        ai_factory_add_cuda_unit(equity ${model}/${product})
    endforeach()
endforeach()

# Temporary side-by-side policy prototypes used by the equivalence and
# performance benchmark. They deliberately keep distinct public launchers.
foreach(model IN ITEMS cev merton)
    ai_factory_add_cuda_unit(equity ${model}/athena_autocallbis)
    target_link_libraries(
        ai_factory_equity_${model}_athena_autocallbis PUBLIC
        ai_factory_product_athena_autocall_dataset
    )
endforeach()

ai_factory_add_cuda_unit(equity bates/athena_autocallbis)
target_link_libraries(
    ai_factory_equity_bates_athena_autocallbis PUBLIC
    ai_factory_product_athena_autocall_dataset
)

foreach(product IN ITEMS asian_option up_and_out_option)
    ai_factory_add_cuda_unit(equity heston/${product}bis)
    target_link_libraries(
        ai_factory_equity_heston_${product}bis PUBLIC
        ai_factory_product_${product}_dataset
    )
endforeach()

ai_factory_add_cuda_unit(equity merton/forward_start_optionbis)
target_link_libraries(
    ai_factory_equity_merton_forward_start_optionbis PUBLIC
    ai_factory_product_forward_start_option_dataset
)

ai_factory_add_cuda_unit(equity variance_gamma/asian_optionbis)
target_link_libraries(
    ai_factory_equity_variance_gamma_asian_optionbis PUBLIC
    ai_factory_product_asian_option_dataset
)

if(AI_FACTORY_MATHDX_ROOT)
    if(NOT EXISTS "${AI_FACTORY_MATHDX_ROOT}/include/cufftdx.hpp")
        message(FATAL_ERROR
            "AI_FACTORY_MATHDX_ROOT does not contain include/cufftdx.hpp"
        )
    endif()
    if(NOT CUDA_WORKBENCH_ARCHITECTURES STREQUAL "89")
        message(FATAL_ERROR
            "The tuned cuFFTDx rough-Bergomi pricer currently targets SM 8.9; "
            "configure CUDA_WORKBENCH_ARCHITECTURES=89"
        )
    endif()
    ai_factory_add_cuda_unit(
        equity rough_bergomi/european_option_fft
    )
    target_include_directories(
        ai_factory_equity_rough_bergomi_european_option_fft PRIVATE
        ${AI_FACTORY_MATHDX_ROOT}/include
        ${AI_FACTORY_MATHDX_ROOT}/external/cutlass/include
    )
    target_compile_definitions(
        ai_factory_equity_rough_bergomi_european_option_fft PUBLIC
        AI_FACTORY_HAS_CUFFTDX=1
    )
endif()

set(_ai_factory_fixed_income_units
    cir/sample
    cir/european_swaption
    cir/rate_option
    cir/zero_coupon_bond_option
    g2/sample
    g2/rate_option
    g2/zero_coupon_bond_option
    g2_plus_plus/sample
    g2_plus_plus/nelson_siegel/rate_option
    g2_plus_plus/nelson_siegel/zero_coupon_bond_option
    g2_plus_plus/svensson/rate_option
    g2_plus_plus/svensson/zero_coupon_bond_option
    hull_white/sample
    hull_white/nelson_siegel/european_swaption
    hull_white/nelson_siegel/rate_option
    hull_white/nelson_siegel/zero_coupon_bond_option
    hull_white/svensson/european_swaption
    hull_white/svensson/rate_option
    hull_white/svensson/zero_coupon_bond_option
    ornstein_uhlenbeck/sample
    ornstein_uhlenbeck/european_swaption
    ornstein_uhlenbeck/rate_option
    ornstein_uhlenbeck/zero_coupon_bond_option
    vasicek/sample
    vasicek/european_swaption
    vasicek/rate_option
    vasicek/zero_coupon_bond_option
)
foreach(unit_path IN LISTS _ai_factory_fixed_income_units)
    ai_factory_add_cuda_unit(fixed_income ${unit_path})
endforeach()

# Aggregates are explicit convenience targets, not dependencies of a single
# generator or test. They are therefore suitable for CI and full-domain work.
add_custom_target(equity)
add_custom_target(fixed_income)
add_custom_target(all_models)
add_custom_target(dataset_loaders)

foreach(model IN LISTS _ai_factory_equity_models)
    get_property(units GLOBAL PROPERTY AI_FACTORY_equity_${model}_CUDA_UNITS)
    add_custom_target(model_${model} DEPENDS ${units})
    add_dependencies(equity model_${model})
endforeach()
foreach(model IN LISTS _ai_factory_fixed_income_models)
    get_property(
        units GLOBAL PROPERTY AI_FACTORY_fixed_income_${model}_CUDA_UNITS
    )
    add_custom_target(model_${model} DEPENDS ${units})
    add_dependencies(fixed_income model_${model})
endforeach()
add_dependencies(all_models equity fixed_income)

get_property(
    _ai_factory_dataset_targets GLOBAL PROPERTY AI_FACTORY_DATASET_TARGETS
)
add_dependencies(dataset_loaders ${_ai_factory_dataset_targets})

# Compatibility umbrella for external consumers that intentionally request
# every launcher. Internal generators and tests never link this target.
get_property(
    _ai_factory_cuda_unit_targets
    GLOBAL PROPERTY AI_FACTORY_CUDA_UNIT_TARGETS
)
add_library(cuda_workbench INTERFACE)
target_link_libraries(cuda_workbench INTERFACE
    ${_ai_factory_cuda_unit_targets}
    ai_factory_dataset_core
)

# Infer exact link dependencies from public project headers included by one
# executable. Missing candidates are inline-only headers such as dynamics or
# analytics and intentionally require no compiled library.
function(ai_factory_collect_source_dependencies output source)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_text)
    string(REGEX MATCHALL
        "#include[ \t]+\"(model|curve|product)/[^\"]+\\.(cuh|hpp)\""
        project_includes
        "${source_text}"
    )
    set(dependencies)
    foreach(include_line IN LISTS project_includes)
        string(REGEX REPLACE
            ".*\"([^\"]+)\\.(cuh|hpp)\".*" "\\1"
            header_path "${include_line}"
        )
        string(REGEX REPLACE "^model/" "" target_path "${header_path}")
        string(REPLACE "/" "_" target_id "${target_path}")
        set(candidate ai_factory_${target_id})
        if(TARGET ${candidate})
            list(APPEND dependencies ${candidate})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES dependencies)
    set(${output} ${dependencies} PARENT_SCOPE)
endfunction()
