// Zero-coupon bond call dataset row and host-side JSON loader.
#pragma once

#include "product/zero_coupon_bond_option/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

// Load every zero-coupon bond option into one contiguous vector.
std::vector<ZeroCouponBondOptionParameters> load_zero_coupon_bond_options(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
