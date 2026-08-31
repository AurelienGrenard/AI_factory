# Performance benchmark targets and architecture experiments.
add_custom_target(performance_benchmarks)

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
