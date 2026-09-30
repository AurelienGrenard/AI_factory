# Host and CUDA contract tests for the selectable price-gradient integration.
add_executable(test_price_gradients_launch_plan EXCLUDE_FROM_ALL tests/price_gradients/launch_plan_test.cpp)
target_include_directories(
    test_price_gradients_launch_plan
    PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(test_price_gradients_launch_plan PRIVATE ai_factory_cuda_tuning nlohmann_json::nlohmann_json)
target_compile_features(test_price_gradients_launch_plan PRIVATE cxx_std_23)
add_dependencies(ai_factory_host_tests test_price_gradients_launch_plan)
add_test(NAME price_gradients_launch_plan COMMAND test_price_gradients_launch_plan)
set_tests_properties(price_gradients_launch_plan PROPERTIES LABELS "workbench;price_gradients;offline" TIMEOUT 30)
add_executable(test_price_gradients_configuration EXCLUDE_FROM_ALL
    tests/price_gradients/configuration_test.cpp)
ai_factory_configure_host_library(test_price_gradients_configuration)
add_dependencies(ai_factory_host_tests test_price_gradients_configuration)
add_test(NAME price_gradients_configuration COMMAND test_price_gradients_configuration)
set_tests_properties(price_gradients_configuration PROPERTIES
    LABELS "workbench;price_gradients;offline" TIMEOUT 30)
add_executable(test_price_gradients_mixed_sensitivity_contract EXCLUDE_FROM_ALL
    tests/price_gradients/mixed_sensitivity_contract_test.cpp)
ai_factory_configure_host_library(test_price_gradients_mixed_sensitivity_contract)
add_dependencies(ai_factory_host_tests test_price_gradients_mixed_sensitivity_contract)
add_test(NAME price_gradients_mixed_sensitivity_contract
    COMMAND test_price_gradients_mixed_sensitivity_contract)
set_tests_properties(price_gradients_mixed_sensitivity_contract PROPERTIES
    LABELS "workbench;price_gradients;offline;mixed_hessian" TIMEOUT 30)
add_executable(test_price_gradients_parameter_domains EXCLUDE_FROM_ALL
    tests/price_gradients/parameter_domain_test.cpp)
ai_factory_configure_host_library(test_price_gradients_parameter_domains)
add_dependencies(ai_factory_host_tests test_price_gradients_parameter_domains)
add_test(NAME price_gradients_parameter_domains
    COMMAND test_price_gradients_parameter_domains)
set_tests_properties(price_gradients_parameter_domains PROPERTIES
    LABELS "workbench;price_gradients;parameters" TIMEOUT 30)
add_cuda_workbench_test(price_gradients_european_cuda
    tests/price_gradients/european_cuda_test.cu "price_gradients;parity" 120)
add_cuda_workbench_test(price_gradients_central_work_cuda
    tests/price_gradients/central_work_cuda_test.cu "price_gradients;work_counts" 60)
add_cuda_workbench_test(price_gradients_terminal_node_graph_cuda
    tests/price_gradients/terminal_node_graph_cuda_test.cu
    "price_gradients;parity;node_graph;heston;cev;merton;bates;jumps" 180)
target_link_libraries(test_price_gradients_terminal_node_graph_cuda PRIVATE
    ai_factory_equity_bates_european_option_price_gradients
    ai_factory_equity_heston_european_option_price_gradients
    ai_factory_equity_cev_european_option_price_gradients
    ai_factory_equity_merton_european_option_price_gradients)
add_cuda_workbench_test(price_gradients_mixed_terminal_node_graph_cuda
    tests/price_gradients/mixed_terminal_node_graph_cuda_test.cu
    "price_gradients;parity;node_graph;heston;mixed_hessian" 180)
target_link_libraries(test_price_gradients_mixed_terminal_node_graph_cuda PRIVATE
    ai_factory_equity_heston_european_option_price_gradients
    ai_factory_equity_merton_european_option_price_gradients)
add_cuda_workbench_test(price_gradients_mixed_path_node_graph_cuda
    tests/price_gradients/mixed_path_node_graph_cuda_test.cu
    "price_gradients;parity;node_graph;path_products;jumps;mixed_hessian" 240)
target_link_libraries(test_price_gradients_mixed_path_node_graph_cuda PRIVATE
    ai_factory_equity_black_scholes_asian_option_price_gradients
    ai_factory_equity_black_scholes_forward_start_option_price_gradients
    ai_factory_equity_black_scholes_geometric_asian_option_price_gradients
    ai_factory_equity_black_scholes_range_accrual_price_gradients
    ai_factory_equity_bates_cliquet_price_gradients
    ai_factory_equity_bates_range_accrual_price_gradients
    ai_factory_equity_heston_asian_option_price_gradients
    ai_factory_equity_merton_forward_start_option_price_gradients)
add_cuda_workbench_test(price_gradients_path_node_graph_cuda
    tests/price_gradients/path_node_graph_cuda_test.cu
    "price_gradients;parity;node_graph;path_products;calendars;jumps" 180)
target_link_libraries(test_price_gradients_path_node_graph_cuda PRIVATE
    ai_factory_equity_black_scholes_asian_option_price_gradients
    ai_factory_equity_bates_range_accrual_price_gradients
    ai_factory_equity_heston_asian_option_price_gradients
    ai_factory_equity_merton_athena_autocall_price_gradients
    ai_factory_equity_merton_forward_start_option_price_gradients)
add_cuda_workbench_test(price_gradients_terminal_binding_matrix_cuda
    tests/price_gradients/terminal_binding_matrix_cuda_test.cpp
    "price_gradients;parity;node_graph;terminal_products;binding_matrix" 300)
add_cuda_workbench_test(price_gradients_fixed_income_terminal_node_graph_cuda
    tests/price_gradients/fixed_income_terminal_node_graph_cuda_test.cpp
    "price_gradients;parity;node_graph;fixed_income;g2" 180)
target_link_libraries(test_price_gradients_fixed_income_terminal_node_graph_cuda PRIVATE
    ai_factory_fixed_income_g2_european_swaption_price_gradients
    ai_factory_fixed_income_g2_plus_plus_flat_european_swaption_price_gradients
    ai_factory_fixed_income_g2_plus_plus_nelson_siegel_european_swaption_price_gradients
    ai_factory_fixed_income_g2_plus_plus_svensson_european_swaption_price_gradients)
add_cuda_workbench_test(price_gradients_cev_cuda
    tests/price_gradients/cev_cuda_test.cu "price_gradients;parity;cev" 120)
add_cuda_workbench_test(price_gradients_sabr_cuda
    tests/price_gradients/sabr_cuda_test.cu "price_gradients;parity;sabr" 120)
add_cuda_workbench_test(price_gradients_diffusion_models_cuda
    tests/price_gradients/diffusion_models_cuda_test.cu
    "price_gradients;parity;heston_3_2;schobel_zhu;stein_stein" 180)
add_cuda_workbench_test(price_gradients_merton_cuda
    tests/price_gradients/merton_cuda_test.cu "price_gradients;parity;merton" 120)
add_cuda_workbench_test(price_gradients_kou_cuda
    tests/price_gradients/kou_cuda_test.cu "price_gradients;parity;kou" 180)
add_cuda_workbench_test(price_gradients_bates_cuda
    tests/price_gradients/bates_cuda_test.cu "price_gradients;parity;bates" 240)
add_cuda_workbench_test(price_gradients_levy_cuda
    tests/price_gradients/levy_cuda_test.cu
    "price_gradients;parity;variance_gamma;normal_inverse_gaussian" 240)
add_cuda_workbench_test(price_gradients_terminal_products_cuda
    tests/price_gradients/terminal_products_cuda_test.cu
    "price_gradients;parity;terminal_products;heston;merton" 180)
add_cuda_workbench_test(price_gradients_diagonal_dataset_cuda
    tests/price_gradients/diagonal_dataset_cuda_test.cpp
    "price_gradients;dataset;heston;cir;fixed_income" 120)
target_link_libraries(test_price_gradients_diagonal_dataset_cuda PRIVATE
    ai_factory_price_gradient_dataset
    ai_factory_equity_heston_american_option_price_gradients
    ai_factory_fixed_income_cir_european_swaption_price_gradients)
add_cuda_workbench_test(price_gradients_american_cuda
    tests/price_gradients/american_cuda_test.cu
    "price_gradients;parity;american_option;longstaff_schwartz;heston" 180)
add_cuda_workbench_test(price_gradients_bates_american_cuda
    tests/price_gradients/bates_american_cuda_test.cu
    "price_gradients;parity;american_option;longstaff_schwartz;bates;jumps" 240)
add_cuda_workbench_test(price_gradients_exact_transition_american_cuda
    tests/price_gradients/exact_transition_american_cuda_test.cu
    "price_gradients;parity;american_option;longstaff_schwartz;exact_transition" 300)
add_cuda_workbench_test(price_gradients_fixed_step_american_cuda
    tests/price_gradients/fixed_step_american_cuda_test.cu
    "price_gradients;parity;american_option;longstaff_schwartz;fixed_step" 240)
add_cuda_workbench_test(price_gradients_cir_european_swaption_cuda
    tests/price_gradients/cir_european_swaption_cuda_test.cu
    "price_gradients;fixed_income;cir;jamshidian" 120)
add_cuda_workbench_test(price_gradients_flat_fixed_income_cuda
    tests/price_gradients/flat_fixed_income_cuda_test.cu
    "price_gradients;fixed_income;flat_curve;closed_form;warp_packing" 60)
target_link_libraries(test_price_gradients_cir_european_swaption_cuda PRIVATE
    ai_factory_fixed_income_cir_rate_option_price_gradients
    ai_factory_fixed_income_cir_zero_coupon_bond_option_price_gradients)
add_cuda_workbench_test(price_gradients_bermudan_swaption_cuda
    tests/price_gradients/bermudan_swaption_cuda_test.cu
    "price_gradients;fixed_income;bermudan_swaption;longstaff_schwartz" 300)
add_offline_stage_test(price_gradients_dataset ai_factory_price_gradient_dataset tests/price_gradients/dataset_test.cpp)
if(Python3_Interpreter_FOUND)
    add_test(NAME price_gradients_artifact_contract COMMAND ${Python3_EXECUTABLE} -m unittest discover
        -s tests/price_gradients -p "test_*.py")
    set_tests_properties(price_gradients_artifact_contract PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} LABELS "workbench;price_gradients;offline;codegen" TIMEOUT 30)
    add_test(NAME price_gradients_performance_manifest
        COMMAND ${Python3_EXECUTABLE} -m unittest
            tests.performance.test_price_gradient_strategy_manifest)
    set_tests_properties(price_gradients_performance_manifest PROPERTIES
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        LABELS "workbench;price_gradients;performance;architecture" TIMEOUT 30)
endif()

# Explicit performance executable; no timing thresholds in routine CTest.
add_executable(benchmark_price_gradients_terminal EXCLUDE_FROM_ALL
    tests/performance/price_gradients/terminal.cu)
target_include_directories(benchmark_price_gradients_terminal PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_price_gradients_terminal PRIVATE ai_factory_runtime
    ai_factory_equity_black_scholes_european_option_price_gradients
    ai_factory_equity_heston_european_option_price_gradients
    ai_factory_equity_merton_european_option_price_gradients)
set_target_properties(benchmark_price_gradients_terminal PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

add_executable(benchmark_price_gradients_closed_form EXCLUDE_FROM_ALL tests/performance/price_gradients/closed_form.cu)
target_include_directories(benchmark_price_gradients_closed_form PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_price_gradients_closed_form PRIVATE ai_factory_runtime
    ai_factory_equity_black_scholes_european_option_price_gradients)
set_target_properties(benchmark_price_gradients_closed_form PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

# Explicit numerical experiment: no stochastic acceptance thresholds in routine CTest.
add_executable(study_price_gradients_bumps EXCLUDE_FROM_ALL tests/price_gradients/bump_study.cu)
target_include_directories(study_price_gradients_bumps PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(study_price_gradients_bumps PRIVATE ai_factory_runtime
    ai_factory_equity_black_scholes_european_option_price_gradients
    ai_factory_equity_heston_european_option_price_gradients
    ai_factory_equity_cev_european_option_price_gradients)
set_target_properties(study_price_gradients_bumps PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

add_executable(benchmark_price_gradients_cev EXCLUDE_FROM_ALL tests/performance/price_gradients/cev.cu)
target_include_directories(benchmark_price_gradients_cev PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_price_gradients_cev PRIVATE ai_factory_runtime
    ai_factory_equity_cev_european_option_price_gradients ai_factory_equity_cev_european_option_price_delta)
set_target_properties(benchmark_price_gradients_cev PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

add_executable(benchmark_price_gradients_sabr EXCLUDE_FROM_ALL
    tests/performance/price_gradients/sabr.cu)
target_include_directories(benchmark_price_gradients_sabr PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_price_gradients_sabr PRIVATE ai_factory_runtime
    ai_factory_equity_sabr_european_option_price_gradients)
set_target_properties(benchmark_price_gradients_sabr PROPERTIES
    CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

add_executable(benchmark_price_gradients_american EXCLUDE_FROM_ALL
    tests/performance/price_gradients/american.cu)
target_include_directories(benchmark_price_gradients_american PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_price_gradients_american PRIVATE ai_factory_runtime
    ai_factory_equity_heston_american_option_price_gradients)
set_target_properties(benchmark_price_gradients_american PROPERTIES
    CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

add_executable(benchmark_price_gradients_jamshidian EXCLUDE_FROM_ALL
    tests/performance/price_gradients/jamshidian.cu)
target_include_directories(benchmark_price_gradients_jamshidian PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(benchmark_price_gradients_jamshidian PRIVATE ai_factory_runtime
    ai_factory_fixed_income_cir_european_swaption_price_gradients)
set_target_properties(benchmark_price_gradients_jamshidian PROPERTIES
    CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)

add_custom_target(price_gradients_performance_benchmarks DEPENDS
    benchmark_price_gradients_terminal
    benchmark_price_gradients_closed_form
    benchmark_price_gradients_cev
    benchmark_price_gradients_american
    benchmark_price_gradients_jamshidian)

add_executable(study_price_gradients_heston_rates EXCLUDE_FROM_ALL tests/price_gradients/heston_rate_study.cu)
target_include_directories(study_price_gradients_heston_rates PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(study_price_gradients_heston_rates PRIVATE ai_factory_runtime)
set_target_properties(study_price_gradients_heston_rates PROPERTIES CUDA_STANDARD 23 CUDA_STANDARD_REQUIRED YES)
