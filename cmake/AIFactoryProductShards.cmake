# Product shards used by published Cartesian Monte Carlo datasets.
# Each source has its own recipe and can be regenerated from its 1000-row parent.
set(AI_FACTORY_PRODUCT_SHARD_GENERATOR_SOURCES
    catalog/product/american_option/american_options_01_shard00/generator.cpp
    catalog/product/asian_option/asian_options_01_shard00/generator.cpp
    catalog/product/asset_or_nothing_option/asset_or_nothing_options_01_shard00/generator.cpp
    catalog/product/athena_autocall/athena_autocalls_01_shard00/generator.cpp
    catalog/product/bermudan_swaption/bermudan_swaptions_01_shard00/generator.cpp
    catalog/product/cliquet/cliquets_01_shard00/generator.cpp
    catalog/product/digital_option/digital_options_01_shard00/generator.cpp
    catalog/product/double_knock_out_option/double_knock_out_options_01_shard00/generator.cpp
    catalog/product/down_and_in_option/down_and_in_options_01_shard00/generator.cpp
    catalog/product/down_and_out_option/down_and_out_options_01_shard00/generator.cpp
    catalog/product/european_option/european_options_01_shard00/generator.cpp
    catalog/product/european_swaption/european_swaptions_01_shard00/generator.cpp
    catalog/product/forward_start_option/forward_start_options_01_shard00/generator.cpp
    catalog/product/gap_option/gap_call_options_01_shard00/generator.cpp
    catalog/product/gap_option/gap_put_options_01_shard00/generator.cpp
    catalog/product/geometric_asian_option/geometric_asian_options_01_shard00/generator.cpp
    catalog/product/lookback_option/lookback_options_01_shard00/generator.cpp
    catalog/product/phoenix_autocall/phoenix_autocalls_01_shard00/generator.cpp
    catalog/product/phoenix_memory_autocall/phoenix_memory_autocalls_01_shard00/generator.cpp
    catalog/product/range_accrual/range_accruals_01_shard00/generator.cpp
    catalog/product/rate_option/rate_options_01_shard00/generator.cpp
    catalog/product/straddle/straddles_01_shard00/generator.cpp
    catalog/product/up_and_in_option/up_and_in_options_01_shard00/generator.cpp
    catalog/product/up_and_out_option/up_and_out_options_01_shard00/generator.cpp
    catalog/product/up_no_touch/up_no_touches_01_shard00/generator.cpp
    catalog/product/up_one_touch/up_one_touches_01_shard00/generator.cpp
    catalog/product/zero_coupon_bond_option/zero_coupon_bond_options_01_shard00/generator.cpp
    catalog/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01_shard00/generator.cpp
)
foreach(source IN LISTS AI_FACTORY_PRODUCT_SHARD_GENERATOR_SOURCES)
    ai_factory_register_catalog_generator("${source}" parameter)
endforeach()
