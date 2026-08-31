# Host, CUDA and architecture test registration.
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

    function(add_offline_stage_test stage library)
        set(target test_${stage}_stage)
        add_executable(
            ${target} EXCLUDE_FROM_ALL tests/${stage}_stage_test.cpp
        )
        target_link_libraries(${target} PRIVATE ${library})
        target_compile_features(${target} PRIVATE cxx_std_23)
        add_dependencies(ai_factory_host_tests ${target})
        add_test(NAME ${stage}_stage COMMAND ${target})
        set_tests_properties(${stage}_stage PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;offline;${stage}"
            TIMEOUT 30
        )
    endfunction()
    add_offline_stage_test(sampling ai_factory_sampling)
    add_offline_stage_test(artifact_io ai_factory_artifact_io)
    add_offline_stage_test(parameter_dataset ai_factory_parameter_dataset)
    add_offline_stage_test(sample_dataset ai_factory_sample_dataset)
    add_offline_stage_test(price_dataset ai_factory_price_dataset)

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

    add_executable(
        test_simulation_schedule_validation
        EXCLUDE_FROM_ALL
        tests/simulation_schedule_validation_test.cpp
    )
    set_source_files_properties(
        tests/simulation_schedule_validation_test.cpp
        PROPERTIES LANGUAGE CUDA
    )
    target_include_directories(
        test_simulation_schedule_validation
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(
        test_simulation_schedule_validation
        PRIVATE ai_factory_runtime
    )
    target_compile_features(
        test_simulation_schedule_validation PRIVATE cxx_std_23
    )
    add_dependencies(
        ai_factory_host_tests test_simulation_schedule_validation
    )
    add_test(
        NAME simulation_schedule_validation
        COMMAND test_simulation_schedule_validation
    )
    set_tests_properties(simulation_schedule_validation PROPERTIES
        LABELS "workbench;simulation;launch_validation"
    )

    add_executable(
        test_rough_sabr_dataset_loader
        EXCLUDE_FROM_ALL
        tests/rough_sabr_dataset_loader_test.cpp
    )
    target_link_libraries(
        test_rough_sabr_dataset_loader
        PRIVATE ai_factory_equity_rough_sabr_dataset
    )
    target_compile_features(test_rough_sabr_dataset_loader PRIVATE cxx_std_23)
    add_dependencies(ai_factory_host_tests test_rough_sabr_dataset_loader)
    add_test(
        NAME rough_sabr_dataset_loader
        COMMAND test_rough_sabr_dataset_loader
    )
    set_tests_properties(rough_sabr_dataset_loader PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;dataset;rough_sabr"
    )


    include(cmake/AIFactoryValidation.cmake)

    # Python-owned architecture and performance contract checks.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME performance_baseline_checker
            COMMAND
                ${Python3_EXECUTABLE} -m unittest
                tests.performance.test_performance_protocol
        )
        set_tests_properties(performance_baseline_checker PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;performance;baseline"
            TIMEOUT 30
        )
        add_test(
            NAME pricing_binding_codegen
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/codegen/pricing_bindings/generate.py
                --family all
                --output
                ${CMAKE_BINARY_DIR}/generated/pricing_bindings
                --compare-root ${CMAKE_SOURCE_DIR}
        )
        set_tests_properties(pricing_binding_codegen PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;codegen;equity"
            TIMEOUT 60
        )
        add_test(
            NAME pricing_capability_manifest
            COMMAND
                ${Python3_EXECUTABLE} -m unittest
                tools.codegen.pricing_bindings.test_capability_manifest
        )
        set_tests_properties(pricing_capability_manifest PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;codegen;architecture"
            TIMEOUT 30
        )
        add_test(
            NAME model_source_layout
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/cuda/check_model_layout.py
        )
        set_tests_properties(model_source_layout PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;codegen;architecture"
            TIMEOUT 30
        )
        add_test(
            NAME catalog_generator_boundaries
            COMMAND
                ${Python3_EXECUTABLE}
                ${CMAKE_SOURCE_DIR}/tools/cuda/check_catalog_generators.py
        )
        set_tests_properties(catalog_generator_boundaries PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;catalog;architecture"
            TIMEOUT 30
        )

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
        numerical_robustness_cuda
        tests/numerical_robustness_cuda_test.cu
        numerical_robustness
        30
    )
    add_cuda_workbench_test(
        monte_carlo_statistics_precision_cuda
        tests/monte_carlo_statistics_precision_cuda_test.cu
        "numerical_robustness;performance"
        60
    )
    add_cuda_workbench_test(
        asian_mean_precision_cuda
        tests/asian_mean_precision_cuda_test.cu
        "numerical_robustness;performance;asian"
        60
    )
    add_cuda_workbench_test(
        range_accrual_sum_precision_cuda
        tests/range_accrual_sum_precision_cuda_test.cu
        "numerical_robustness;performance;range_accrual"
        60
    )
    add_cuda_workbench_test(
        fractional_resolvent_precision_cuda
        tests/fractional_resolvent_precision_cuda_test.cu
        "numerical_robustness;performance;volterra;rough_stein_stein"
        120
    )
    add_cuda_workbench_test(
        cuda_pricing_runner
        tests/cuda_pricing_runner_test.cu
        "offline;pricing_runner"
        30
    )
    add_cuda_workbench_test(
        longstaff_schwartz_regressor_cuda
        tests/longstaff_schwartz_regressor_cuda_test.cu
        longstaff_schwartz
        30
    )
    target_link_libraries(
        test_longstaff_schwartz_regressor_cuda PRIVATE
        ai_factory_longstaff_schwartz
    )
    add_cuda_workbench_test(
        longstaff_schwartz_precision_cuda
        tests/longstaff_schwartz_precision_cuda_test.cu
        "longstaff_schwartz;numerical_robustness;performance"
        60
    )
    target_link_libraries(
        test_longstaff_schwartz_precision_cuda PRIVATE
        ai_factory_longstaff_schwartz
    )
    add_cuda_workbench_test(
        path_product_factorization_cuda
        tests/path_product_factorization_cuda_test.cu
        "factorization;monte_carlo;products"
        120
    )
    add_cuda_workbench_test(
        bermudan_swaption_cuda
        tests/bermudan_swaption_cuda_test.cpp
        "longstaff_schwartz;bermudan_swaption"
        120
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
        fixed_income_analytics_contract_cuda
        tests/fixed_income_analytics_contract_cuda_test.cu
        analytics_contract
        30
    )
    add_cuda_workbench_test(
        fixed_income_dynamics_policy_cuda
        tests/fixed_income_dynamics_policy_cuda_test.cu
        dynamics_policy
        30
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
        equity_dynamics_policy_cuda
        tests/equity_dynamics_policy_cuda_test.cu
        dynamics_policy
        30
    )
    add_cuda_workbench_test(
        levy_dynamics_cuda
        tests/levy_dynamics_cuda_test.cu
        levy_dynamics
        30
    )
    add_cuda_workbench_test(
        merton_kou_cev_schobel_zhu_dynamics_cuda
        tests/merton_kou_cev_schobel_zhu_dynamics_cuda_test.cu
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
        sabr_absorbing_boundary_cuda
        tests/sabr_absorbing_boundary_cuda_test.cpp
        "sabr;absorbing_boundary"
        60
    )
    add_cuda_workbench_test(
        volterra_kernel_policy_cuda
        tests/volterra_kernel_policy_cuda_test.cu
        "volterra;kernel_policy"
        30
    )
    if(AI_FACTORY_MATHDX_ROOT)
        add_cuda_workbench_test(
            rough_bergomi_european_option_cuda
            tests/rough_bergomi_european_option_cuda_test.cpp
            "rough_bergomi_products;rough_sabr_products"
            120
        )
        add_cuda_workbench_test(
            rough_volterra_product_policy_cuda
            tests/rough_volterra_product_policy_cuda_test.cu
            "rough_bergomi_products;product_policy"
            120
        )
        add_cuda_workbench_test(
            rough_volterra_samples_cuda
            tests/rough_volterra_samples_cuda_test.cpp
            "samples;rough_bergomi;rough_sabr"
            120
        )
        target_link_libraries(
            test_rough_volterra_product_policy_cuda PRIVATE
            ai_factory_cufftdx
        )
    endif()
    add_cuda_workbench_test(
        rough_heston_european_option_cuda
        tests/rough_heston_european_option_cuda_test.cu
        rough_heston_products
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_european_option_cuda
        tests/quadratic_rough_heston_european_option_cuda_test.cu
        "rough_heston_products;quadratic_rough_heston;numerical_robustness"
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_asian_options_cuda
        tests/quadratic_rough_heston_asian_options_cuda_test.cu
        "rough_heston_products;quadratic_rough_heston;asian;numerical_robustness"
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_preparation_cuda
        tests/quadratic_rough_heston_preparation_cuda_test.cu
        "rough_heston;preparation"
        120
    )
    add_cuda_workbench_test(
        volterra_fft_workspace_bounds_cuda
        tests/volterra_fft_workspace_bounds_cuda_test.cu
        "volterra;workspace"
        30
    )
    add_cuda_workbench_test(
        rough_heston_samples_cuda
        tests/rough_heston_samples_cuda_test.cu
        "samples;rough_heston"
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_samples_cuda
        tests/quadratic_rough_heston_samples_cuda_test.cu
        "samples;quadratic_rough_heston;numerical_robustness"
        120
    )
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
        levy_american_option_cuda
        tests/levy_american_option_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        black_scholes_cev_kou_merton_schobel_zhu_american_lsm_cuda
        tests/black_scholes_cev_kou_merton_schobel_zhu_american_lsm_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        cuda_kernel_diagnostics
        tests/cuda_kernel_diagnostics_test.cu
        kernel_diagnostics
        30
    )
    add_cuda_workbench_test(
        policy_size_budgets_cuda
        tests/policy_size_budgets_cuda_test.cu
        "kernel_diagnostics;policy_budgets"
        30
    )

endif()
