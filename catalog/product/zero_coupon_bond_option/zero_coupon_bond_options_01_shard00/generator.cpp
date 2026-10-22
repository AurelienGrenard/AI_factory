// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{5, 13, 24, 36, 45, 50, 64, 77, 85, 96, 104, 114, 124, 139, 147, 158, 162, 173, 185, 195, 204, 211, 227, 239, 243, 256, 269, 274, 289, 293, 304, 310, 328, 337, 347, 354, 361, 377, 386, 392, 405, 418, 420, 438, 448, 456, 469, 472, 483, 490, 509, 514, 520, 532, 544, 558, 568, 573, 586, 595, 601, 611, 623, 632, 641, 653, 661, 671, 685, 692, 705, 718, 720, 730, 740, 753, 763, 777, 780, 797, 807, 810, 825, 834, 843, 858, 861, 877, 885, 890, 903, 916, 922, 933, 942, 959, 964, 973, 986, 995};
    write_product_shard(
        "datasets/product/zero_coupon_bond_option/zero_coupon_bond_options_01.json",
        "datasets/product/zero_coupon_bond_option/zero_coupon_bond_options_01_shard00.json",
        "catalog/product/zero_coupon_bond_option/zero_coupon_bond_options_01_shard00/generation.yaml",
        selected_rows);
}
