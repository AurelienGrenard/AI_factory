# Joint-model pricing is a distinct composition domain.
ai_factory_add_dataset_library(
    ai_factory_correlation_dataset src/correlation/dataset.cpp
)
ai_factory_add_dataset_library(
    ai_factory_product_worst_of_call_dataset
    src/product/worst_of_call/dataset.cpp
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

add_executable(generate_correlation_matrices_01 EXCLUDE_FROM_ALL
    catalog/correlation/correlation_matrices_01/generator.cpp
)
target_link_libraries(generate_correlation_matrices_01 PRIVATE
    ai_factory_artifact_io ai_factory_correlation_dataset
)

set_source_files_properties(
    catalog/multi_asset/worst_of_call/bs2_worst_of_calls_01/generator.cpp
    PROPERTIES LANGUAGE CUDA
)
add_executable(generate_bs2_worst_of_calls_01 EXCLUDE_FROM_ALL
    catalog/multi_asset/worst_of_call/bs2_worst_of_calls_01/generator.cpp
)
target_include_directories(generate_bs2_worst_of_calls_01 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(generate_bs2_worst_of_calls_01 PRIVATE
    ai_factory_equity_black_scholes_dataset
    ai_factory_product_worst_of_call_dataset
    ai_factory_correlation_dataset
    ai_factory_multi_asset_price_dataset
    ai_factory_cuda_tuning
)
set_target_properties(generate_bs2_worst_of_calls_01 PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
)
add_dependencies(price_generators generate_bs2_worst_of_calls_01)

add_executable(generate_bs_rough_heston_matrices_01 EXCLUDE_FROM_ALL
    catalog/correlation/bs_rough_heston_matrices_01/generator.cpp
)
target_link_libraries(generate_bs_rough_heston_matrices_01 PRIVATE
    ai_factory_artifact_io ai_factory_correlation_dataset
)

set_source_files_properties(
    catalog/multi_asset/worst_of_call/bs_rough_heston_worst_of_calls_01/generator.cpp
    PROPERTIES LANGUAGE CUDA
)
add_executable(generate_bs_rough_heston_worst_of_calls_01 EXCLUDE_FROM_ALL
    catalog/multi_asset/worst_of_call/bs_rough_heston_worst_of_calls_01/generator.cpp
)
target_include_directories(generate_bs_rough_heston_worst_of_calls_01 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(generate_bs_rough_heston_worst_of_calls_01 PRIVATE
    ai_factory_equity_black_scholes_dataset
    ai_factory_equity_rough_heston_dataset
    ai_factory_product_worst_of_call_dataset
    ai_factory_correlation_dataset
    ai_factory_multi_asset_price_dataset
    ai_factory_causal_fft
)
set_target_properties(generate_bs_rough_heston_worst_of_calls_01 PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
)
add_dependencies(price_generators generate_bs_rough_heston_worst_of_calls_01)

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

add_executable(generate_bs_heston_vasicek_matrices_01 EXCLUDE_FROM_ALL
    catalog/correlation/bs_heston_vasicek_matrices_01/generator.cpp
)
target_link_libraries(generate_bs_heston_vasicek_matrices_01 PRIVATE
    ai_factory_artifact_io ai_factory_correlation_dataset
)

set_source_files_properties(
    catalog/multi_asset/worst_of_call/bs_heston_vasicek_worst_of_calls_01/generator.cpp
    PROPERTIES LANGUAGE CUDA
)
add_executable(generate_bs_heston_vasicek_worst_of_calls_01 EXCLUDE_FROM_ALL
    catalog/multi_asset/worst_of_call/bs_heston_vasicek_worst_of_calls_01/generator.cpp
)
target_include_directories(generate_bs_heston_vasicek_worst_of_calls_01 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(generate_bs_heston_vasicek_worst_of_calls_01 PRIVATE
    ai_factory_equity_black_scholes_dataset
    ai_factory_equity_heston_dataset
    ai_factory_fixed_income_vasicek_dataset
    ai_factory_product_worst_of_call_dataset
    ai_factory_correlation_dataset
    ai_factory_multi_asset_price_dataset
    ai_factory_causal_fft
)
set_target_properties(generate_bs_heston_vasicek_worst_of_calls_01 PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES
)
add_dependencies(price_generators generate_bs_heston_vasicek_worst_of_calls_01)
