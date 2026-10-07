#pragma once

#include "product/zero_coupon_bond_up_and_out/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

std::vector<ZeroCouponBondUpAndOutParameters> load_zero_coupon_bond_up_and_outs(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
