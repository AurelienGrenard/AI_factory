# Fine-grained build graph for CUDA pricing units and JSON dataset loaders.
#
# Numerical implementation files such as dynamics_impl.cuh and analytics_impl.cuh remain
# force-inlined includes of their consuming kernels. Only public launch units
# are compiled here, so no relocatable-device-code boundary is introduced.

# This checked-in fragment is generated beside every equity pricing binding;
# CI compares both the C++ units and this registration matrix to one manifest.
include(cmake/generated/EquityPricingBindings.cmake)

function(ai_factory_configure_host_library target)
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    # Public headers expose CUDA Runtime types. Host-only consumers must see
    # the headers belonging to the same toolkit as CMAKE_CUDA_COMPILER; the
    # distribution's /usr/include/cuda_runtime.h may describe an older ABI.
    target_include_directories(${target} SYSTEM PUBLIC
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
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
    target_link_libraries(${target} PUBLIC ai_factory_dataset_validation)
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
    bermudan_swaption
    european_swaption
    rate_option
    zero_coupon_bond_option
    ${AI_FACTORY_GENERATED_EQUITY_PRODUCTS}
)
list(REMOVE_DUPLICATES _ai_factory_products)
foreach(product IN LISTS _ai_factory_products)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/product/${product}/dataset.cpp")
        ai_factory_add_dataset_library(
            ai_factory_product_${product}_dataset
            src/product/${product}/dataset.cpp
        )
    endif()
endforeach()

set(_ai_factory_equity_models ${AI_FACTORY_GENERATED_EQUITY_MODELS})
set(_ai_factory_fixed_income_models
    cir
    g2
    g2_plus_plus
    hull_white
    ornstein_uhlenbeck
    vasicek
)

function(ai_factory_equity_model_family output model)
    if(model IN_LIST AI_FACTORY_GENERATED_ROUGH_MODELS)
        set(${output} rough PARENT_SCOPE)
    else()
        set(${output} markovian PARENT_SCOPE)
    endif()
endfunction()

foreach(model IN LISTS _ai_factory_equity_models)
    ai_factory_equity_model_family(model_family ${model})
    ai_factory_add_dataset_library(
        ai_factory_equity_${model}_dataset
        src/model/equity/${model_family}/${model}/dataset.cpp
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
    string(REGEX REPLACE "^([^/]+).*$" "\\1" model "${unit_path}")
    if(domain STREQUAL "equity")
        ai_factory_equity_model_family(model_family ${model})
        set(source src/model/equity/${model_family}/${unit_path}.cu)
    else()
        set(source src/model/${domain}/${unit_path}.cu)
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
        return()
    endif()

    add_library(${target} STATIC EXCLUDE_FROM_ALL ${source})
    ai_factory_configure_cuda_library(${target})

    get_filename_component(product "${unit_path}" NAME)
    set(dependencies
        ai_factory_runtime
        ai_factory_${domain}_${model}_dataset
    )
    if(TARGET ai_factory_product_${product}_dataset)
        list(APPEND dependencies ai_factory_product_${product}_dataset)
    endif()
    if(unit_path MATCHES "^[^/]+/(nelson_siegel|svensson)/")
        list(APPEND dependencies
            ai_factory_curve_${CMAKE_MATCH_1}_dataset
        )
    endif()
    if(product STREQUAL "american_option"
        OR product STREQUAL "bermudan_swaption")
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

foreach(unit_path IN LISTS AI_FACTORY_GENERATED_EQUITY_REGULAR_UNITS)
    ai_factory_add_cuda_unit(equity ${unit_path})
endforeach()

# Samples and American exercise are deliberately outside the generated
# non-American product matrix. Existence is the source of truth for these
# exceptional units, so no second model-product list is maintained.
foreach(model IN LISTS _ai_factory_equity_models)
    if(NOT model IN_LIST AI_FACTORY_GENERATED_VOLTERRA_MODELS)
        ai_factory_add_cuda_unit(equity ${model}/sample)
    endif()
    ai_factory_add_cuda_unit(equity ${model}/american_option)
endforeach()

if(AI_FACTORY_MATHDX_ROOT)
    if(NOT EXISTS "${AI_FACTORY_MATHDX_ROOT}/include/cufftdx.hpp")
        message(FATAL_ERROR
            "AI_FACTORY_MATHDX_ROOT does not contain include/cufftdx.hpp"
        )
    endif()
    if(NOT CUDA_WORKBENCH_ARCHITECTURES STREQUAL "89")
        message(FATAL_ERROR
            "The tuned cuFFTDx Volterra pricers currently target SM 8.9; "
            "configure CUDA_WORKBENCH_ARCHITECTURES=89"
        )
    endif()
    set(_ai_factory_rough_fft_targets)
    foreach(unit_path IN LISTS AI_FACTORY_GENERATED_EQUITY_VOLTERRA_UNITS)
        ai_factory_add_cuda_unit(equity ${unit_path})
        string(REPLACE "/" "_" unit_id "${unit_path}")
        list(APPEND _ai_factory_rough_fft_targets
            ai_factory_equity_${unit_id}
        )
    endforeach()
    foreach(model IN LISTS AI_FACTORY_GENERATED_VOLTERRA_MODELS)
        if(EXISTS
            "${CMAKE_CURRENT_SOURCE_DIR}/src/model/equity/rough/${model}/sample.cu"
        )
            ai_factory_add_cuda_unit(equity ${model}/sample)
            list(APPEND _ai_factory_rough_fft_targets
                ai_factory_equity_${model}_sample
            )
        endif()
    endforeach()
    foreach(target IN LISTS _ai_factory_rough_fft_targets)
        target_include_directories(
            ${target} PRIVATE
            ${AI_FACTORY_MATHDX_ROOT}/include
            ${AI_FACTORY_MATHDX_ROOT}/external/cutlass/include
        )
        target_compile_definitions(
            ${target} PUBLIC AI_FACTORY_HAS_CUFFTDX=1
        )
    endforeach()
endif()

set(_ai_factory_fixed_income_units
    cir/bermudan_swaption
    cir/sample
    cir/european_swaption
    cir/rate_option
    cir/zero_coupon_bond_option
    g2/sample
    g2/bermudan_swaption
    g2/rate_option
    g2/zero_coupon_bond_option
    g2_plus_plus/sample
    g2_plus_plus/nelson_siegel/bermudan_swaption
    g2_plus_plus/nelson_siegel/rate_option
    g2_plus_plus/nelson_siegel/zero_coupon_bond_option
    g2_plus_plus/svensson/rate_option
    g2_plus_plus/svensson/zero_coupon_bond_option
    g2_plus_plus/svensson/bermudan_swaption
    hull_white/sample
    hull_white/nelson_siegel/bermudan_swaption
    hull_white/nelson_siegel/european_swaption
    hull_white/nelson_siegel/rate_option
    hull_white/nelson_siegel/zero_coupon_bond_option
    hull_white/svensson/european_swaption
    hull_white/svensson/rate_option
    hull_white/svensson/zero_coupon_bond_option
    hull_white/svensson/bermudan_swaption
    ornstein_uhlenbeck/bermudan_swaption
    ornstein_uhlenbeck/sample
    ornstein_uhlenbeck/european_swaption
    ornstein_uhlenbeck/rate_option
    ornstein_uhlenbeck/zero_coupon_bond_option
    vasicek/sample
    vasicek/bermudan_swaption
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
    ai_factory_dataset_validation
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
        # The equity family folders organize source files but deliberately do
        # not participate in stable target names.
        string(REGEX REPLACE
            "^model/equity/(markovian|rough)/"
            "model/equity/"
            header_path
            "${header_path}"
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
