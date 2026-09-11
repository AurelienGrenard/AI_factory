# Performance benchmark targets and architecture experiments.
add_custom_target(performance_benchmarks)

# Opt-in Jamshidian geometry study; no catalogue or production tuning mutation.
add_executable(ai_factory_jamshidian_strategy_benchmark EXCLUDE_FROM_ALL
    tests/performance/jamshidian_strategy_benchmark.cu
)
ai_factory_configure_cuda_library(ai_factory_jamshidian_strategy_benchmark)
target_link_libraries(ai_factory_jamshidian_strategy_benchmark PRIVATE
    ai_factory_runtime
    ai_factory_fixed_income_cir_dataset
    ai_factory_fixed_income_cir_plus_plus_dataset
    ai_factory_fixed_income_ornstein_uhlenbeck_dataset
    ai_factory_fixed_income_vasicek_dataset
    ai_factory_fixed_income_hull_white_dataset
    ai_factory_curve_nelson_siegel_dataset
    ai_factory_curve_svensson_dataset
    ai_factory_product_european_swaption_dataset
)

# Opt-in scaling probes call production bindings; LSM runs as a separate campaign.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    set(_scaling_arguments --output "${CMAKE_BINARY_DIR}/pricing-scaling")
    if(AI_FACTORY_MATHDX_ROOT)
        list(APPEND _scaling_arguments --mathdx)
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/tools/performance/pricing_scaling_manifest.py"
            ${_scaling_arguments}
        RESULT_VARIABLE _scaling_result
        OUTPUT_VARIABLE _scaling_output
        ERROR_VARIABLE _scaling_error
    )
    if(NOT _scaling_result EQUAL 0)
        message(FATAL_ERROR "Pricing scaling generation failed: ${_scaling_error}")
    endif()
    add_custom_target(pricing_scaling_benchmarks)
    include("${CMAKE_BINARY_DIR}/pricing-scaling/targets.cmake")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/tools/performance/pricing_scaling_manifest.py"
        "${CMAKE_SOURCE_DIR}/tests/performance/pricing_scaling/benchmark.cu.tpl"
        "${CMAKE_SOURCE_DIR}/tests/performance/pricing_scaling/n_factor_inputs.cuh.tpl"
        "${CMAKE_SOURCE_DIR}/tests/performance/pricing_scaling/volterra_fft_inputs.cuh.tpl"
        "${CMAKE_SOURCE_DIR}/tests/performance/pricing_scaling/curve_inputs.cuh.tpl"
    )
endif()

# Explicit exploratory target, outside the protocol-v3 baseline workloads.
add_executable(ai_factory_cir_discount_chain_probe EXCLUDE_FROM_ALL
    tests/performance/cir_forward_measure/discount_chain_probe.cu
)
ai_factory_configure_cuda_library(ai_factory_cir_discount_chain_probe)
target_link_libraries(ai_factory_cir_discount_chain_probe PRIVATE
    ai_factory_runtime ai_factory_fixed_income_cir_dataset
)

add_executable(ai_factory_cir_forward_transition_probe EXCLUDE_FROM_ALL
    tests/performance/cir_forward_measure/transition_probe.cu
)
ai_factory_configure_cuda_library(ai_factory_cir_forward_transition_probe)
target_link_libraries(ai_factory_cir_forward_transition_probe PRIVATE
    ai_factory_runtime ai_factory_fixed_income_cir_dataset
)

add_executable(ai_factory_cir_forward_measure_probe EXCLUDE_FROM_ALL
    tests/performance/cir_forward_measure/benchmark.cu
)
ai_factory_configure_cuda_library(ai_factory_cir_forward_measure_probe)
target_link_libraries(ai_factory_cir_forward_measure_probe PRIVATE
    ai_factory_fixed_income_cir_bermudan_swaption
    ai_factory_fixed_income_cir_dataset
    ai_factory_product_bermudan_swaption_dataset
)

add_executable(ai_factory_fixed_income_lsm_probe EXCLUDE_FROM_ALL
    tests/performance/fixed_income_lsm_probe.cu
)
ai_factory_configure_cuda_library(ai_factory_fixed_income_lsm_probe)
target_link_libraries(ai_factory_fixed_income_lsm_probe PRIVATE
    ai_factory_fixed_income_cir_bermudan_swaption
    ai_factory_fixed_income_ornstein_uhlenbeck_bermudan_swaption
    ai_factory_fixed_income_g2_bermudan_swaption
    ai_factory_fixed_income_cir_dataset
    ai_factory_fixed_income_ornstein_uhlenbeck_dataset
    ai_factory_fixed_income_g2_dataset
    ai_factory_product_bermudan_swaption_dataset
    ai_factory_fixed_income_vasicek_bermudan_swaption
    ai_factory_fixed_income_vasicek_dataset
    ai_factory_fixed_income_hull_white_nelson_siegel_bermudan_swaption
    ai_factory_fixed_income_hull_white_svensson_bermudan_swaption
    ai_factory_fixed_income_g2_plus_plus_nelson_siegel_bermudan_swaption
    ai_factory_fixed_income_g2_plus_plus_svensson_bermudan_swaption
    ai_factory_equity_black_scholes_american_option
    ai_factory_equity_heston_american_option
    ai_factory_equity_bates_american_option
    ai_factory_equity_variance_gamma_american_option
    ai_factory_product_american_option_dataset
    ai_factory_fixed_income_hull_white_dataset
    ai_factory_fixed_income_g2_plus_plus_dataset
    ai_factory_curve_nelson_siegel_dataset
    ai_factory_curve_svensson_dataset
    ai_factory_equity_black_scholes_dataset
    ai_factory_equity_heston_dataset
    ai_factory_equity_bates_dataset
    ai_factory_equity_variance_gamma_dataset
)

add_executable(
    ai_factory_generic_kernel_benchmark EXCLUDE_FROM_ALL
    tests/performance/generic_kernel_benchmark.cu
)
ai_factory_configure_cuda_library(ai_factory_generic_kernel_benchmark)
target_link_libraries(
    ai_factory_generic_kernel_benchmark PRIVATE
    ai_factory_runtime
    ai_factory_equity_black_scholes_european_option
    ai_factory_equity_bates_phoenix_autocall
    ai_factory_equity_bates_phoenix_memory_autocall
    ai_factory_fixed_income_hull_white_nelson_siegel_european_swaption
)
add_dependencies(
    performance_benchmarks ai_factory_generic_kernel_benchmark
)

add_executable(
    ai_factory_early_exercise_benchmark EXCLUDE_FROM_ALL
    tests/performance/early_exercise_benchmark.cu
)
ai_factory_configure_cuda_library(ai_factory_early_exercise_benchmark)
target_link_libraries(
    ai_factory_early_exercise_benchmark PRIVATE
    ai_factory_runtime
    ai_factory_equity_black_scholes_american_option
    ai_factory_equity_heston_american_option
    ai_factory_fixed_income_ornstein_uhlenbeck_bermudan_swaption
    ai_factory_fixed_income_g2_bermudan_swaption
)
add_dependencies(
    performance_benchmarks ai_factory_early_exercise_benchmark
)

add_library(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment
    STATIC EXCLUDE_FROM_ALL
    src/model/fixed_income/cir/product/european_swaption.cu
)
ai_factory_configure_cuda_library(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment
)
target_compile_definitions(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment PRIVATE
    AI_FACTORY_NCX2_FORCE_INLINE=1
)
target_link_libraries(
    ai_factory_fixed_income_cir_european_swaption_inline_experiment PUBLIC
    ai_factory_runtime
    ai_factory_fixed_income_cir_dataset
    ai_factory_product_european_swaption_dataset
)

foreach(variant IN ITEMS inline noinline)
    set(target ai_factory_cir_kernel_benchmark_${variant})
    add_executable(
        ${target} EXCLUDE_FROM_ALL
        tests/performance/cir_kernel_benchmark.cu
    )
    ai_factory_configure_cuda_library(${target})
    target_compile_definitions(
        ${target} PRIVATE
        AI_FACTORY_CIR_BENCHMARK_VARIANT="${variant}"
    )
    if(variant STREQUAL "inline")
        target_link_libraries(
            ${target} PRIVATE
            ai_factory_fixed_income_cir_european_swaption_inline_experiment
        )
    else()
        target_link_libraries(
            ${target} PRIVATE
            ai_factory_fixed_income_cir_european_swaption
        )
    endif()
    add_dependencies(performance_benchmarks ${target})
endforeach()

if(AI_FACTORY_MATHDX_ROOT)
    add_executable(
        ai_factory_model_sample_benchmark EXCLUDE_FROM_ALL
        tests/performance/model_sample_benchmark.cu
    )
    ai_factory_configure_cuda_library(ai_factory_model_sample_benchmark)
    target_link_libraries(
        ai_factory_model_sample_benchmark PRIVATE
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_black_scholes_sample
        ai_factory_equity_heston_sample
        ai_factory_equity_rough_heston_sample
        ai_factory_equity_rough_bergomi_sample
    )
    add_dependencies(
        performance_benchmarks ai_factory_model_sample_benchmark
    )

    add_library(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        STATIC EXCLUDE_FROM_ALL
        src/model/equity/rough/rough_bergomi/product/european_option.cu
    )
    ai_factory_configure_cuda_library(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
    )
    target_compile_definitions(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        PRIVATE AI_FACTORY_VOLTERRA_DIRECT_MAX_STEP_COUNT=32
    )
    target_link_libraries(
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        PUBLIC
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_rough_bergomi_dataset
        ai_factory_product_european_option_dataset
    )

    add_executable(
        ai_factory_volterra_kernel_benchmark EXCLUDE_FROM_ALL
        tests/performance/volterra_kernel_benchmark.cu
    )
    ai_factory_configure_cuda_library(ai_factory_volterra_kernel_benchmark)
    target_link_libraries(
        ai_factory_volterra_kernel_benchmark PRIVATE
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_heston_european_option
        ai_factory_equity_rough_heston_european_option
        ai_factory_equity_rough_bergomi_european_option
        ai_factory_equity_rough_sabr_european_option
    )
    add_dependencies(
        performance_benchmarks ai_factory_volterra_kernel_benchmark
    )

    add_executable(
        ai_factory_volterra_direct_kernel_benchmark EXCLUDE_FROM_ALL
        tests/performance/volterra_kernel_benchmark.cu
    )
    ai_factory_configure_cuda_library(
        ai_factory_volterra_direct_kernel_benchmark
    )
    target_compile_definitions(
        ai_factory_volterra_direct_kernel_benchmark PRIVATE
        AI_FACTORY_VOLTERRA_BENCHMARK_VARIANT="direct_experiment"
    )
    target_link_libraries(
        ai_factory_volterra_direct_kernel_benchmark PRIVATE
        ai_factory_runtime
        ai_factory_cufftdx
        ai_factory_equity_heston_european_option
        ai_factory_equity_rough_heston_european_option
        ai_factory_equity_rough_bergomi_european_option_direct_experiment
        ai_factory_equity_rough_sabr_european_option
    )
    add_dependencies(
        performance_benchmarks ai_factory_volterra_direct_kernel_benchmark
    )
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(AI_FACTORY_MATHDX_ROOT AND Python3_Interpreter_FOUND)
    add_custom_target(
        performance_regression_gate
        COMMAND
            ${Python3_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tools/performance/run_baseline.py
            --baseline
            ${CMAKE_SOURCE_DIR}/tests/performance/baseline_sm89_v3.json
            --build-dir ${CMAKE_BINARY_DIR}
            --output
            ${CMAKE_BINARY_DIR}/performance_candidate_sm89_v3.ndjson
        DEPENDS performance_benchmarks
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        USES_TERMINAL
    )
endif()
