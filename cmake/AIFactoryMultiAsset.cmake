# Joint-model pricing is a distinct composition domain.
ai_factory_add_dataset_library(
    ai_factory_correlation_dataset src/correlation/dataset.cpp
)
ai_factory_add_dataset_library(
    ai_factory_product_worst_of_call_dataset
    src/product/worst_of_call/dataset.cpp
)
ai_factory_add_dataset_library(
    ai_factory_product_best_of_call_dataset
    src/product/best_of_call/dataset.cpp
)
ai_factory_add_dataset_library(
    ai_factory_product_basket_call_dataset
    src/product/basket_call/dataset.cpp
)
ai_factory_add_offline_library(
    ai_factory_multi_asset_price_dataset
    tools/datasets/multi_asset_price_dataset.cpp
)
target_link_libraries(ai_factory_multi_asset_price_dataset PUBLIC
    ai_factory_artifact_io ai_factory_dataset_validation
)

add_executable(generate_worst_of_calls_01 EXCLUDE_FROM_ALL
    catalog/product/worst_of_call/worst_of_calls_01/generator.cpp
)
target_link_libraries(generate_worst_of_calls_01 PRIVATE
    ai_factory_parameter_dataset ai_factory_dataset_validation
)
add_dependencies(parameter_generators generate_worst_of_calls_01)

# Factorized scenarios reuse model loaders and the existing correlated kernels.
ai_factory_add_dataset_library(
    ai_factory_fixed_income_constant_rate_dataset
    src/model/fixed_income/constant_rate/dataset.cpp
)
add_executable(generate_constant_rates_01 EXCLUDE_FROM_ALL
    catalog/model/fixed_income/constant_rate/parameters/constant_rates_01/generator.cpp
)
target_link_libraries(generate_constant_rates_01 PRIVATE
    ai_factory_parameter_dataset ai_factory_dataset_validation
)
add_dependencies(parameter_generators generate_constant_rates_01)

# Each leaf owns its recipe and tiny generator; one loop registers all targets.
foreach(factor_model IN ITEMS
    equity/markovian/bates
    equity/markovian/black_scholes
    equity/markovian/cev
    equity/markovian/heston
    equity/markovian/heston_3_2
    equity/markovian/kou
    equity/markovian/merton
    equity/markovian/normal_inverse_gaussian
    equity/markovian/sabr
    equity/markovian/schobel_zhu
    equity/markovian/stein_stein
    equity/markovian/variance_gamma
    equity/rough/log_modulated_rough_bergomi
    equity/rough/quadratic_rough_heston
    equity/rough/rough_bergomi
    equity/rough/rough_heston
    equity/rough/rough_sabr
    equity/rough/rough_stein_stein
    fixed_income/g2
    fixed_income/g2_plus_plus
)
    get_filename_component(model_name "${factor_model}" NAME)
    set(factor_target generate_${model_name}_factors_01)
    add_executable(${factor_target} EXCLUDE_FROM_ALL
        catalog/model/${factor_model}/factor_parameters/${model_name}_factors_01/generator.cpp)
    target_link_libraries(${factor_target} PRIVATE
        ai_factory_parameter_dataset ai_factory_dataset_validation)
    add_dependencies(parameter_generators ${factor_target})
endforeach()

# Core-only factor bases and terminal basket contracts for the 11-factor study.
foreach(core_model IN ITEMS
    equity/markovian/heston
    equity/markovian/bates
    equity/markovian/sabr
    equity/rough/rough_heston
    equity/rough/rough_bergomi
    fixed_income/cir
)
    get_filename_component(model_name "${core_model}" NAME)
    set(core_target generate_${model_name}_multi_core_01)
    add_executable(${core_target} EXCLUDE_FROM_ALL
        catalog/model/${core_model}/factor_parameters/${model_name}_multi_core_01/generator.cpp)
    target_link_libraries(${core_target} PRIVATE
        ai_factory_parameter_dataset ai_factory_dataset_validation)
    add_dependencies(parameter_generators ${core_target})
endforeach()

foreach(basket_family IN ITEMS worst_of_call best_of_call basket_call)
    if(basket_family STREQUAL "worst_of_call")
        set(basket_dataset worst_of_calls_multi_01)
    elseif(basket_family STREQUAL "best_of_call")
        set(basket_dataset best_of_calls_multi_01)
    else()
        set(basket_dataset basket_calls_multi_01)
    endif()
    set(basket_target generate_${basket_dataset})
    add_executable(${basket_target} EXCLUDE_FROM_ALL
        catalog/product/${basket_family}/${basket_dataset}/generator.cpp)
    target_link_libraries(${basket_target} PRIVATE
        ai_factory_parameter_dataset ai_factory_dataset_validation)
    add_dependencies(parameter_generators ${basket_target})
endforeach()

add_executable(generate_five_equity_cir_correlations_01 EXCLUDE_FROM_ALL
    catalog/correlation/five_equity_cir_01/generator.cpp)
target_link_libraries(generate_five_equity_cir_correlations_01 PRIVATE
    ai_factory_correlation_dataset ai_factory_artifact_io)
add_dependencies(parameter_generators
    generate_five_equity_cir_correlations_01)

# Local experiments are present only in a developer work directory.
if(EXISTS "${CMAKE_SOURCE_DIR}/work/cmake/AIFactoryMultiAssetExperiments.cmake")
    include("${CMAKE_SOURCE_DIR}/work/cmake/AIFactoryMultiAssetExperiments.cmake")
endif()
