# Catalog generator registration and dependency discovery.
# Register one host-only parameter generator with shared build settings.
add_custom_target(parameter_generators)
add_custom_target(price_generators)
add_custom_target(sample_generators)

function(ai_factory_collect_generation_dependencies output source)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_text)
    string(REGEX MATCHALL
        "tools/datasets/[a-z0-9_]+_generation\\.hpp"
        generation_headers
        "${source_text}"
    )
    set(dependencies)
    foreach(header IN LISTS generation_headers)
        string(REGEX REPLACE
            "tools/datasets/([^/]+)\\.hpp" "ai_factory_\\1"
            candidate "${header}"
        )
        if(TARGET ${candidate})
            list(APPEND dependencies ${candidate})
        endif()
    endforeach()
    string(REGEX MATCHALL
        "model/equity/(markovian|rough)/[a-z0-9_]+/dataset\\.hpp"
        equity_model_dataset_headers
        "${source_text}"
    )
    foreach(header IN LISTS equity_model_dataset_headers)
        string(REGEX REPLACE
            "model/equity/(markovian|rough)/([^/]+)/dataset\\.hpp"
            "ai_factory_equity_\\2_dataset"
            candidate
            "${header}"
        )
        if(TARGET ${candidate})
            list(APPEND dependencies ${candidate})
        endif()
    endforeach()
    string(REGEX MATCHALL
        "model/fixed_income/[a-z0-9_]+/dataset\\.hpp"
        fixed_income_model_dataset_headers
        "${source_text}"
    )
    foreach(header IN LISTS fixed_income_model_dataset_headers)
        string(REGEX REPLACE
            "model/fixed_income/([^/]+)/dataset\\.hpp"
            "ai_factory_fixed_income_\\1_dataset"
            candidate
            "${header}"
        )
        if(TARGET ${candidate})
            list(APPEND dependencies ${candidate})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES dependencies)
    set(${output} ${dependencies} PARENT_SCOPE)
endfunction()

function(add_parameter_generator target source)
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    ai_factory_collect_generation_dependencies(dependencies ${source})
    target_link_libraries(
        ${target} PRIVATE
        ai_factory_parameter_dataset
        ai_factory_dataset_validation
        ${dependencies}
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    add_dependencies(parameter_generators ${target})
endfunction()

# Register one CUDA price generator with shared build settings.
function(add_price_generator target source)
    if(source MATCHES
        "catalog/model/equity/rough/(rough_heston|quadratic_rough_heston)/prices/")
        # Host preparation includes the CUDA dynamics contract that defines
        # PreparedDynamics<N>; compile these generated recipes with nvcc even
        # though their entry-point extension remains generator.cpp.
        set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    ai_factory_collect_source_dependencies(dependencies ${source})
    if(NOT dependencies)
        message(FATAL_ERROR
            "No compiled pricing unit found for generator ${target}: ${source}"
        )
    endif()
    target_link_libraries(
        ${target} PRIVATE ${dependencies} ai_factory_price_dataset
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED YES
    )
    add_dependencies(price_generators ${target})
endfunction()

# Register one model-sample generator against its thin CUDA model binding.
function(add_sample_generator target source)
    if(source MATCHES
        "catalog/model/equity/rough/(rough_heston|quadratic_rough_heston)/samples/")
        # Host N-factor preparation includes CUDA-decorated dynamics types.
        set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(${target} EXCLUDE_FROM_ALL ${source})
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    ai_factory_collect_source_dependencies(dependencies ${source})
    if(NOT dependencies)
        message(FATAL_ERROR
            "No compiled sample unit found for generator ${target}: ${source}"
        )
    endif()
    target_link_libraries(
        ${target} PRIVATE ${dependencies} ai_factory_sample_dataset
    )
    target_compile_features(${target} PRIVATE cxx_std_23)
    set_target_properties(${target} PROPERTIES
        CUDA_STANDARD 23
        CUDA_STANDARD_REQUIRED YES
    )
    add_dependencies(sample_generators ${target})
endfunction()

# Derive a stable executable name from the versioned catalog recipe id. For
# example `heston_01__european_calls_01__01` remains the public target
# `generate_heston_european_calls_01`; fitted recipes include their curve id.
function(ai_factory_catalog_generator_target output source)
    get_filename_component(recipe_directory "${source}" DIRECTORY)
    get_filename_component(recipe_id "${recipe_directory}" NAME)
    if(source MATCHES "/prices/")
        string(REPLACE "__" ";" components "${recipe_id}")
        list(POP_BACK components version)
        set(target generate)
        foreach(component IN LISTS components)
            string(REGEX REPLACE "_[0-9]+$" "" component "${component}")
            string(APPEND target "_${component}")
        endforeach()
        string(APPEND target "_${version}")
    elseif(source MATCHES
        "catalog/model/equity/(markovian|rough)/([^/]+)/samples/")
        set(target "generate_${CMAKE_MATCH_2}_${recipe_id}")
    elseif(source MATCHES
        "catalog/model/fixed_income/([^/]+)/samples/")
        set(target "generate_${CMAKE_MATCH_1}_${recipe_id}")
    else()
        set(target "generate_${recipe_id}")
    endif()
    set(${output} "${target}" PARENT_SCOPE)
endfunction()

# The generated manifest owns the equity recipes and their implementation
# matrix. Tree discovery registers fixed-income and parameter recipes while
# the capability checker verifies their declared matrix.
file(GLOB_RECURSE _ai_factory_catalog_generators
    CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/catalog/*/generator.cpp"
)
list(SORT _ai_factory_catalog_generators)
foreach(source IN LISTS _ai_factory_catalog_generators)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" source_text)
    if(NOT AI_FACTORY_MATHDX_ROOT
        AND source_text MATCHES
            "model/equity/rough/(rough_bergomi|rough_sabr|log_modulated_rough_bergomi|rough_stein_stein)/")
        continue()
    endif()
    ai_factory_catalog_generator_target(target "${source}")
    if(TARGET ${target})
        message(FATAL_ERROR
            "Duplicate catalog generator target ${target}: ${source}"
        )
    endif()
    if(source MATCHES "/prices/")
        add_price_generator(${target} "${source}")
    elseif(source MATCHES "/samples/")
        add_sample_generator(${target} "${source}")
    else()
        add_parameter_generator(${target} "${source}")
    endif()
endforeach()
