# Independent QuantLib/Premia and immutable cached-reference checks.
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
                add_test(
                    NAME quantlib_bermudan_swaption_adapter
                    COMMAND
                        ${Python3_EXECUTABLE} -m unittest
                        validation.quantlib.tests.test_bermudan_swaption
                )
                set_tests_properties(
                    quantlib_bermudan_swaption_adapter
                    PROPERTIES
                        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                        LABELS "workbench;validation;quantlib;bermudan_swaption"
                )
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
                    datasets/model/equity/markovian/heston/prices/european_calls/heston_01__european_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_european_put
                    validation.quantlib.model.equity.heston.european_put
                    datasets/model/equity/markovian/heston/prices/european_puts/heston_01__european_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_straddle
                    validation.quantlib.model.equity.heston.straddle
                    datasets/model/equity/markovian/heston/prices/straddles/heston_01__straddles_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_digital_call
                    validation.quantlib.model.equity.heston.digital_call
                    datasets/model/equity/markovian/heston/prices/digital_calls/heston_01__digital_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_digital_put
                    validation.quantlib.model.equity.heston.digital_put
                    datasets/model/equity/markovian/heston/prices/digital_puts/heston_01__digital_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_asset_or_nothing_call
                    validation.quantlib.model.equity.heston.asset_or_nothing_call
                    datasets/model/equity/markovian/heston/prices/asset_or_nothing_calls/heston_01__asset_or_nothing_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_asset_or_nothing_put
                    validation.quantlib.model.equity.heston.asset_or_nothing_put
                    datasets/model/equity/markovian/heston/prices/asset_or_nothing_puts/heston_01__asset_or_nothing_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_gap_call
                    validation.quantlib.model.equity.heston.gap_call
                    datasets/model/equity/markovian/heston/prices/gap_calls/heston_01__gap_calls_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    heston_gap_put
                    validation.quantlib.model.equity.heston.gap_put
                    datasets/model/equity/markovian/heston/prices/gap_puts/heston_01__gap_puts_01__01.json
                    heston
                )
                add_quantlib_validation_test(
                    cev_european_call
                    validation.quantlib.model.equity.cev.european_call
                    datasets/model/equity/markovian/cev/prices/european_calls/cev_01__european_calls_01__01.json
                    cev
                )
                add_quantlib_validation_test(
                    cev_european_put
                    validation.quantlib.model.equity.cev.european_put
                    datasets/model/equity/markovian/cev/prices/european_puts/cev_01__european_puts_01__01.json
                    cev
                )

                function(add_bates_validation_test product folder)
                    add_quantlib_validation_test(
                        bates_${product}
                        validation.quantlib.model.equity.bates.${product}
                        datasets/model/equity/markovian/bates/prices/${folder}/bates_01__${folder}_01__01.json
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
                            datasets/model/equity/markovian/variance_gamma/prices/${folder}/variance_gamma_01__${folder}_01__01.json
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
                        datasets/model/equity/markovian/heston/prices/geometric_asian_calls/heston_01__geometric_asian_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_geometric_asian_put
                        validation.quantlib.model.equity.heston.geometric_asian_put
                        datasets/model/equity/markovian/heston/prices/geometric_asian_puts/heston_01__geometric_asian_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_asian_call
                        validation.quantlib.model.equity.heston.asian_call
                        datasets/model/equity/markovian/heston/prices/asian_calls/heston_01__asian_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_asian_put
                        validation.quantlib.model.equity.heston.asian_put
                        datasets/model/equity/markovian/heston/prices/asian_puts/heston_01__asian_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_and_out_call
                        validation.quantlib.model.equity.heston.up_and_out_call
                        datasets/model/equity/markovian/heston/prices/up_and_out_calls/heston_01__up_and_out_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_down_and_out_put
                        validation.quantlib.model.equity.heston.down_and_out_put
                        datasets/model/equity/markovian/heston/prices/down_and_out_puts/heston_01__down_and_out_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_and_in_call
                        validation.quantlib.model.equity.heston.up_and_in_call
                        datasets/model/equity/markovian/heston/prices/up_and_in_calls/heston_01__up_and_in_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_down_and_in_put
                        validation.quantlib.model.equity.heston.down_and_in_put
                        datasets/model/equity/markovian/heston/prices/down_and_in_puts/heston_01__down_and_in_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_one_touch
                        validation.quantlib.model.equity.heston.up_one_touch
                        datasets/model/equity/markovian/heston/prices/up_one_touches/heston_01__up_one_touches_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_up_no_touch
                        validation.quantlib.model.equity.heston.up_no_touch
                        datasets/model/equity/markovian/heston/prices/up_no_touches/heston_01__up_no_touches_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_double_knock_out_call
                        validation.quantlib.model.equity.heston.double_knock_out_call
                        datasets/model/equity/markovian/heston/prices/double_knock_out_calls/heston_01__double_knock_out_calls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_double_knock_out_put
                        validation.quantlib.model.equity.heston.double_knock_out_put
                        datasets/model/equity/markovian/heston/prices/double_knock_out_puts/heston_01__double_knock_out_puts_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_athena_autocall
                        validation.quantlib.model.equity.heston.athena_autocall
                        datasets/model/equity/markovian/heston/prices/athena_autocalls/heston_01__athena_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_phoenix_autocall
                        validation.quantlib.model.equity.heston.phoenix_autocall
                        datasets/model/equity/markovian/heston/prices/phoenix_autocalls/heston_01__phoenix_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_phoenix_memory_autocall
                        validation.quantlib.model.equity.heston.phoenix_memory_autocall
                        datasets/model/equity/markovian/heston/prices/phoenix_memory_autocalls/heston_01__phoenix_memory_autocalls_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_cliquet
                        validation.quantlib.model.equity.heston.cliquet
                        datasets/model/equity/markovian/heston/prices/cliquets/heston_01__cliquets_01__01.json
                        heston
                    )
                    add_quantlib_validation_test(
                        heston_range_accrual
                        validation.quantlib.model.equity.heston.range_accrual
                        datasets/model/equity/markovian/heston/prices/range_accruals/heston_01__range_accruals_01__01.json
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
                        ${CMAKE_SOURCE_DIR}/datasets/model/equity/markovian/${model}/prices/${folder}/${model}_01__${folder}_01__01.json
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


    # Cached publication checks never rerun Premia, QuantLib, or Wine.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        function(add_cached_black_scholes_validation product folder)
            add_test(
                NAME validation_black_scholes_${product}
                COMMAND
                    ${Python3_EXECUTABLE} -m
                    validation.model.equity.black_scholes.${product}
                    ${CMAKE_SOURCE_DIR}/datasets/model/equity/markovian/black_scholes/prices/${folder}/black_scholes_01__${folder}_01__01.json
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
