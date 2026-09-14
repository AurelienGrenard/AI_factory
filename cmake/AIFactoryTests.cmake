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


    add_executable(test_pricing_launch_plan EXCLUDE_FROM_ALL tests/cuda/pricing_launch_plan_test.cpp)
    target_include_directories(test_pricing_launch_plan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_pricing_launch_plan PRIVATE ai_factory_cuda_tuning nlohmann_json::nlohmann_json)
    target_compile_features(test_pricing_launch_plan PRIVATE cxx_std_23)
    add_dependencies(ai_factory_host_tests test_pricing_launch_plan)
    add_test(NAME pricing_launch_plan COMMAND test_pricing_launch_plan)
    set_tests_properties(pricing_launch_plan PROPERTIES LABELS "workbench;offline;cuda-planning" TIMEOUT 30)

    add_dependencies(ai_factory_host_tests inspect_pricing_launch_plan)
    foreach(rough_model IN ITEMS rough_heston rough_sabr)
        add_test(NAME pricing_launch_plan_${rough_model}_delta
            COMMAND inspect_pricing_launch_plan ${rough_model}/european_option 2 --price-delta)
        set_tests_properties(pricing_launch_plan_${rough_model}_delta PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;offline;cuda-planning;price_delta" TIMEOUT 30)
    endforeach()

    add_executable(
        test_dataset_catalog
        EXCLUDE_FROM_ALL
        tests/datasets/catalog_contract_test.cpp
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

    function(add_offline_stage_test stage library source)
        set(target test_${stage}_stage)
        add_executable(
            ${target} EXCLUDE_FROM_ALL ${source}
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
    add_offline_stage_test(sampling ai_factory_sampling
        tests/datasets/parameter_row_sampling_test.cpp)
    add_offline_stage_test(artifact_io ai_factory_artifact_io
        tests/datasets/artifact_io_stage_test.cpp)
    add_offline_stage_test(parameter_dataset ai_factory_parameter_dataset
        tests/datasets/parameter_dataset_stage_test.cpp)
    add_offline_stage_test(sample_dataset ai_factory_sample_dataset
        tests/datasets/sample_dataset_stage_test.cpp)
    add_offline_stage_test(price_dataset ai_factory_price_dataset
        tests/datasets/price_dataset_stage_test.cpp)
    add_offline_stage_test(price_delta_dataset ai_factory_price_dataset
        tests/datasets/price_delta_dataset_stage_test.cpp)

    add_executable(test_sample_host_memory EXCLUDE_FROM_ALL
        tests/sampling/host_memory_test.cpp)
    target_include_directories(test_sample_host_memory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(test_sample_host_memory PRIVATE cxx_std_23)
    add_dependencies(ai_factory_host_tests test_sample_host_memory)
    add_test(NAME sample_host_memory COMMAND test_sample_host_memory)
    set_tests_properties(sample_host_memory PROPERTIES
        LABELS "workbench;offline;sampling" TIMEOUT 30)

    add_executable(
        test_dataset_loaders EXCLUDE_FROM_ALL tests/datasets/dataset_loaders_test.cpp
    )
    ai_factory_collect_source_dependencies(
        dataset_loader_dependencies tests/datasets/dataset_loaders_test.cpp
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
        tests/cuda/simulation_schedule_validation_test.cpp
    )
    set_source_files_properties(
        tests/cuda/simulation_schedule_validation_test.cpp
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
        tests/datasets/rough_sabr_dataset_loader_test.cpp
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
    # Python-owned architecture and performance contract checks.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME cmake_inferred_dependencies
            COMMAND ${Python3_EXECUTABLE} -B
                ${CMAKE_SOURCE_DIR}/tests/build/inferred_dependencies_test.py)
        set_tests_properties(cmake_inferred_dependencies PROPERTIES
            LABELS "workbench;build" TIMEOUT 120)
        foreach(stage IN ITEMS artifact_publication catalog_generation dataset_provenance)
            add_test(NAME ${stage}
                COMMAND ${Python3_EXECUTABLE} -m unittest discover
                    -s tests/datasets -p test_${stage}.py)
            set_tests_properties(${stage} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                LABELS "workbench;offline;generation" TIMEOUT 30)
        endforeach()
        add_test(NAME sample_campaign_selection
            COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s tests/datasets -p test_sample_campaign_selection.py)
        set_tests_properties(sample_campaign_selection PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;offline;generation" TIMEOUT 30)
        add_test(NAME learning_terminal_samples
            COMMAND ${Python3_EXECUTABLE} -m unittest discover
                -s tests/learning -p test_terminal_samples.py)
        set_tests_properties(learning_terminal_samples PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;offline;learning" TIMEOUT 30)
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
            NAME pricing_scaling_protocol
            COMMAND
                ${Python3_EXECUTABLE} -m unittest
                tools.performance.test_pricing_scaling
        )
        set_tests_properties(pricing_scaling_protocol PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            LABELS "workbench;performance;scaling"
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
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        )
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
        philox_cuda tests/numerical/philox_cuda_test.cu philox 30
    )
    add_cuda_workbench_test(
        price_delta_rough_n_factor_cuda
        tests/price_delta/monte_carlo/rough_n_factor_cuda_test.cu
        "price_delta;rough;n_factor" 120
    )
    add_cuda_workbench_test(
        price_delta_catalogue_parity_cuda
        tests/price_delta/catalogue/public_launcher_parity_cuda_test.cu
        "price_delta;catalogue" 120
    )
    add_cuda_workbench_test(
        price_delta_central_parity_cuda
        tests/price_delta/monte_carlo/central_price_parity_cuda_test.cu
        "price_delta;monte_carlo" 60
    )
    add_cuda_workbench_test(
        price_delta_closed_form_cuda
        tests/price_delta/closed_form/european_delta_cuda_test.cu
        "price_delta;closed_form" 30
    )
    add_cuda_workbench_test(
        price_delta_frozen_exercise_cuda
        tests/price_delta/longstaff_schwartz/frozen_exercise_cuda_test.cu
        "price_delta;longstaff_schwartz" 60
    )
    add_cuda_workbench_test(
        price_delta_frozen_path_replay_cuda
        tests/price_delta/longstaff_schwartz/frozen_path_replay_cuda_test.cu
        "price_delta;longstaff_schwartz" 30
    )
    target_link_libraries(test_price_delta_frozen_path_replay_cuda PRIVATE ai_factory_longstaff_schwartz)
    add_cuda_workbench_test(
        numerical_robustness_cuda
        tests/numerical/numerical_robustness_cuda_test.cu
        numerical_robustness
        30
    )
    add_cuda_workbench_test(
        monte_carlo_statistics_precision_cuda
        tests/numerical/monte_carlo_statistics_precision_cuda_test.cu
        "numerical_robustness;performance"
        60
    )
    add_cuda_workbench_test(
        constant_payoff_moments_cuda
        tests/numerical/constant_payoff_moments_cuda_test.cu
        "numerical_robustness;price_delta" 30
    )
    add_cuda_workbench_test(
        asian_mean_precision_cuda
        tests/product/asian_mean_precision_cuda_test.cu
        "numerical_robustness;performance;asian"
        60
    )
    add_cuda_workbench_test(
        geometric_asian_boundary_cuda
        tests/product/geometric_asian_boundary_cuda_test.cu
        "numerical_robustness;asian" 30
    )
    add_cuda_workbench_test(
        volterra_hybrid_schedule_cuda tests/volterra/hybrid_schedule_cuda_test.cu
        "volterra;schedule;numerical_robustness" 30
    )
    add_cuda_workbench_test(
        range_accrual_sum_precision_cuda
        tests/product/range_accrual_sum_precision_cuda_test.cu
        "numerical_robustness;performance;range_accrual"
        60
    )
    add_cuda_workbench_test(
        fractional_resolvent_precision_cuda
        tests/numerical/fractional_resolvent_precision_cuda_test.cu
        "numerical_robustness;performance;volterra;rough_stein_stein"
        120
    )
    add_cuda_workbench_test(
        cuda_pricing_runner
        tests/cuda/cuda_pricing_runner_test.cu
        "offline;pricing_runner"
        30
    )
    add_cuda_workbench_test(
        longstaff_schwartz_regressor_cuda
        tests/longstaff_schwartz/basis_and_solver_cuda_test.cu
        longstaff_schwartz
        30
    )
    target_link_libraries(
        test_longstaff_schwartz_regressor_cuda PRIVATE
        ai_factory_longstaff_schwartz
    )
    add_cuda_workbench_test(
        longstaff_schwartz_precision_cuda
        tests/longstaff_schwartz/precision_cuda_test.cu
        "longstaff_schwartz;numerical_robustness;performance"
        60
    )
    target_link_libraries(
        test_longstaff_schwartz_precision_cuda PRIVATE
        ai_factory_longstaff_schwartz
    )
    add_cuda_workbench_test(
        longstaff_schwartz_normal_residual_cuda
        tests/longstaff_schwartz/normal_residual_cuda_test.cu
        "longstaff_schwartz;numerical_robustness"
        60
    )
    add_cuda_workbench_test(
        kou_lsm_geometry_cuda
        tests/longstaff_schwartz/kou_geometry_cuda_test.cpp
        "longstaff_schwartz;kou;numerical_robustness"
        180
    )
    add_cuda_workbench_test(
        path_product_factorization_cuda
        tests/product/path_product_factorization_cuda_test.cu
        "factorization;monte_carlo;products"
        120
    )
    add_cuda_workbench_test(
        bermudan_swaption_cuda
        tests/longstaff_schwartz/bermudan_swaption_cuda_test.cpp
        "longstaff_schwartz;bermudan_swaption"
        120
    )
    add_cuda_workbench_test(
        noncentral_chi_square_cuda
        tests/numerical/noncentral_chi_square_cuda_test.cu
        distributions
        30
    )
    add_cuda_workbench_test(
        g2_european_swaption_cuda
        tests/model/fixed_income/g2_european_swaption_cuda_test.cpp
        "european_swaption;monte_carlo;numerical_robustness"
        180
    )
    add_cuda_workbench_test(
        cir_cuda tests/model/fixed_income/cir_cuda_test.cu cir 30
    )
    add_cuda_workbench_test(
        cir_plus_plus_cuda tests/model/fixed_income/cir_plus_plus_cuda_test.cu
        "cir_plus_plus;curve_fitting;longstaff_schwartz" 180
    )
    add_cuda_workbench_test(
        fixed_income_analytics_contract_cuda
        tests/model/fixed_income/fixed_income_analytics_contract_cuda_test.cu
        analytics_contract
        30
    )
    add_cuda_workbench_test(
        fixed_income_dynamics_policy_cuda
        tests/model/fixed_income/fixed_income_dynamics_policy_cuda_test.cu
        dynamics_policy
        30
    )
    add_cuda_workbench_test(
        cir_rate_options_cuda
        tests/model/fixed_income/cir_rate_options_cuda_test.cu
        "cir;rate_options"
        30
    )
    add_cuda_workbench_test(
        hull_white_cuda tests/model/fixed_income/hull_white_cuda_test.cu hull_white 30
    )
    add_cuda_workbench_test(
        g2_cuda tests/model/fixed_income/g2_cuda_test.cu g2 30
    )
    add_cuda_workbench_test(
        g2_covariance_cuda tests/model/fixed_income/g2_covariance_cuda_test.cu
        "g2;numerical_robustness" 60
    )
    add_cuda_workbench_test(
        g2_caplet_cuda tests/model/fixed_income/g2_caplet_cuda_test.cu caplet 30
    )
    add_cuda_workbench_test(
        g2_rate_options_cuda
        tests/model/fixed_income/g2_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_cuda
        tests/model/fixed_income/g2_plus_plus_cuda_test.cu
        g2_plus_plus
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_caplet_cuda
        tests/model/fixed_income/g2_plus_plus_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_svensson_caplet_cuda
        tests/model/fixed_income/g2_plus_plus_svensson_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        g2_plus_plus_rate_options_cuda
        tests/model/fixed_income/g2_plus_plus_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        hull_white_caplet_cuda
        tests/model/fixed_income/hull_white_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        hull_white_svensson_caplet_cuda
        tests/model/fixed_income/hull_white_svensson_caplet_cuda_test.cpp
        caplet
        30
    )
    add_cuda_workbench_test(
        hull_white_rate_options_cuda
        tests/model/fixed_income/hull_white_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_caplet_cuda
        tests/model/fixed_income/ornstein_uhlenbeck_caplet_cuda_test.cu
        caplet
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_rate_options_cuda
        tests/model/fixed_income/ornstein_uhlenbeck_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_european_swaptions_cuda
        tests/model/fixed_income/ornstein_uhlenbeck_european_swaptions_cuda_test.cu
        european_swaptions
        30
    )
    add_cuda_workbench_test(
        one_factor_european_swaptions_cuda
        tests/model/fixed_income/one_factor_european_swaptions_cuda_test.cu
        european_swaptions
        30
    )
    add_cuda_workbench_test(
        ornstein_uhlenbeck_cuda
        tests/model/fixed_income/ornstein_uhlenbeck_cuda_test.cu
        ornstein_uhlenbeck
        30
    )
    add_cuda_workbench_test(
        vasicek_cuda tests/model/fixed_income/vasicek_cuda_test.cu vasicek 30
    )
    add_cuda_workbench_test(
        vasicek_caplet_cuda tests/model/fixed_income/vasicek_caplet_cuda_test.cu caplet 30
    )
    add_cuda_workbench_test(
        vasicek_rate_options_cuda
        tests/model/fixed_income/vasicek_rate_options_cuda_test.cu
        rate_options
        30
    )
    add_cuda_workbench_test(
        heston_path_products_cuda
        tests/model/equity/heston_path_products_cuda_test.cpp
        heston_products
        60
    )
    add_cuda_workbench_test(
        heston_terminal_payoffs_cuda
        tests/model/equity/heston_terminal_payoffs_cuda_test.cpp
        heston_products
        60
    )
    add_cuda_workbench_test(
        heston_american_option_cuda
        tests/longstaff_schwartz/heston_american_option_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        bates_dynamics_cuda
        tests/model/equity/bates_dynamics_cuda_test.cu
        bates_dynamics
        30
    )
    add_cuda_workbench_test(
        equity_dynamics_policy_cuda
        tests/model/equity/equity_dynamics_policy_cuda_test.cu
        dynamics_policy
        30
    )
    add_cuda_workbench_test(
        levy_dynamics_cuda
        tests/model/equity/levy_dynamics_cuda_test.cu
        levy_dynamics
        30
    )
    add_cuda_workbench_test(
        merton_kou_cev_schobel_zhu_dynamics_cuda
        tests/model/equity/jump_and_diffusion_dynamics_cuda_test.cu
        equity_dynamics
        30
    )
    add_cuda_workbench_test(
        rough_bergomi_dynamics_cuda
        tests/model/equity/rough/gaussian_volterra_dynamics_cuda_test.cu
        rough_bergomi_dynamics
        30
    )
    add_cuda_workbench_test(
        sabr_absorbing_boundary_cuda
        tests/model/equity/sabr_absorbing_boundary_cuda_test.cpp
        "sabr;absorbing_boundary"
        60
    )
    add_cuda_workbench_test(
        volterra_kernel_policy_cuda
        tests/volterra/volterra_kernel_policy_cuda_test.cu
        "volterra;kernel_policy"
        30
    )
    if(AI_FACTORY_MATHDX_ROOT)
        add_cuda_workbench_test(
            price_delta_rough_fft_path_oracle_cuda tests/price_delta/monte_carlo/rough_fft_path_oracle_cuda_test.cu
            "price_delta;rough;volterra;paired_moments" 60
        )
        target_link_libraries(test_price_delta_rough_fft_path_oracle_cuda PRIVATE ai_factory_cufftdx)
        add_cuda_workbench_test(
            price_delta_rough_fft_cuda tests/price_delta/monte_carlo/rough_fft_cuda_test.cpp
            "price_delta;rough;volterra" 120
        )
        add_cuda_workbench_test(
            rough_bergomi_european_option_cuda
            tests/model/equity/rough/gaussian_volterra_european_option_cuda_test.cpp
            "rough_bergomi_products;rough_sabr_products"
            120
        )
        add_cuda_workbench_test(
            rough_volterra_product_policy_cuda
            tests/volterra/rough_volterra_product_policy_cuda_test.cu
            "rough_bergomi_products;product_policy"
            120
        )
        add_cuda_workbench_test(
            rough_volterra_samples_cuda
            tests/sampling/gaussian_volterra_fft_samples_cuda_test.cpp
            "samples;rough_bergomi;rough_sabr"
            120
        )
        add_cuda_workbench_test(
            sample_recipe_replay_cuda tests/sampling/recipe_replay_cuda_test.cpp
            "samples;codegen;replay" 120
        )
        target_link_libraries(test_sample_recipe_replay_cuda PRIVATE ai_factory_sample_dataset)
        target_link_libraries(
            test_rough_volterra_product_policy_cuda PRIVATE
            ai_factory_cufftdx
        )
    endif()
    add_cuda_workbench_test(
        rough_heston_european_option_cuda
        tests/model/equity/rough/rough_heston_european_option_cuda_test.cu
        rough_heston_products
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_european_option_cuda
        tests/model/equity/rough/quadratic_rough_heston_european_option_cuda_test.cu
        "rough_heston_products;quadratic_rough_heston;numerical_robustness"
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_asian_options_cuda
        tests/model/equity/rough/quadratic_rough_heston_asian_options_cuda_test.cu
        "rough_heston_products;quadratic_rough_heston;asian;numerical_robustness"
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_preparation_cuda
        tests/model/equity/rough/quadratic_rough_heston_preparation_cuda_test.cu
        "rough_heston;preparation"
        120
    )
    add_cuda_workbench_test(
        volterra_fft_workspace_bounds_cuda
        tests/volterra/volterra_fft_workspace_bounds_cuda_test.cu
        "volterra;workspace"
        30
    )
    add_cuda_workbench_test(
        rough_heston_samples_cuda
        tests/sampling/rough_heston_samples_cuda_test.cu
        "samples;rough_heston"
        120
    )
    add_cuda_workbench_test(
        quadratic_rough_heston_samples_cuda
        tests/sampling/quadratic_rough_heston_samples_cuda_test.cu
        "samples;quadratic_rough_heston;numerical_robustness"
        120
    )
    add_cuda_workbench_test(
        black_scholes_cuda
        tests/model/equity/black_scholes_cuda_test.cu
        black_scholes
        30
    )
    add_cuda_workbench_test(
        black_scholes_samples_cuda
        tests/sampling/model_sampling_contract_cuda_test.cu
        samples
        30
    )
    add_cuda_workbench_test(
        bates_path_products_cuda
        tests/model/equity/bates_path_products_cuda_test.cpp
        bates_products
        60
    )
    add_cuda_workbench_test(
        bates_terminal_payoffs_cuda
        tests/model/equity/bates_terminal_payoffs_cuda_test.cpp
        bates_products
        60
    )
    add_cuda_workbench_test(
        bates_american_option_cuda
        tests/longstaff_schwartz/bates_american_option_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        levy_american_option_cuda
        tests/longstaff_schwartz/levy_american_option_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        black_scholes_cev_kou_merton_schobel_zhu_american_lsm_cuda
        tests/longstaff_schwartz/equity_composition_cuda_test.cpp
        american_option
        120
    )
    add_cuda_workbench_test(
        cuda_kernel_diagnostics
        tests/cuda/cuda_kernel_diagnostics_test.cu
        kernel_diagnostics
        30
    )
    add_cuda_workbench_test(
        policy_size_budgets_cuda
        tests/cuda/policy_size_budgets_cuda_test.cu
        "kernel_diagnostics;policy_budgets"
        30
    )

    # Mark the complete main test inventory before the separately owned
    # validation module is included. The standard preset selects this exact
    # set rather than relying on a negative filter.
    get_property(_ai_factory_registered_tests DIRECTORY PROPERTY TESTS)
    foreach(test_name IN LISTS _ai_factory_registered_tests)
        get_property(test_labels TEST ${test_name} PROPERTY LABELS)
        if(NOT validation IN_LIST test_labels)
            set_property(TEST ${test_name} APPEND PROPERTY LABELS main)
        endif()
    endforeach()

endif()
