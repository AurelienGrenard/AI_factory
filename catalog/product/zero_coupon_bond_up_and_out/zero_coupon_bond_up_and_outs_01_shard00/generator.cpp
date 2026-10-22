// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{7, 17, 21, 34, 41, 59, 61, 73, 89, 99, 106, 110, 123, 133, 144, 156, 165, 174, 184, 197, 204, 213, 220, 234, 244, 255, 260, 275, 284, 297, 307, 318, 322, 339, 340, 359, 367, 375, 389, 395, 401, 419, 422, 435, 443, 459, 460, 476, 489, 492, 505, 518, 526, 538, 540, 552, 565, 573, 584, 594, 600, 619, 624, 636, 647, 655, 668, 672, 687, 695, 700, 711, 725, 734, 747, 756, 768, 774, 783, 792, 800, 815, 823, 838, 849, 855, 866, 871, 885, 895, 909, 919, 922, 936, 949, 951, 966, 971, 982, 991};
    write_product_shard(
        "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01.json",
        "datasets/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01_shard00.json",
        "catalog/product/zero_coupon_bond_up_and_out/zero_coupon_bond_up_and_outs_01_shard00/generation.yaml",
        selected_rows);
}
