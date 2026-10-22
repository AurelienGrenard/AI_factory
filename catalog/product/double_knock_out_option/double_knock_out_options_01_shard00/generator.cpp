// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{6, 16, 21, 37, 46, 57, 61, 70, 80, 95, 106, 114, 125, 136, 149, 159, 168, 176, 180, 193, 205, 213, 228, 231, 243, 254, 264, 275, 287, 296, 303, 312, 322, 332, 343, 355, 361, 375, 381, 398, 404, 416, 423, 432, 440, 454, 461, 472, 481, 497, 504, 518, 523, 534, 546, 552, 566, 576, 589, 593, 604, 613, 620, 634, 644, 653, 669, 676, 687, 690, 703, 719, 725, 730, 747, 752, 760, 770, 784, 791, 803, 818, 824, 834, 848, 858, 868, 876, 884, 899, 904, 910, 923, 936, 943, 953, 960, 979, 987, 998};
    write_product_shard(
        "datasets/product/double_knock_out_option/double_knock_out_options_01.json",
        "datasets/product/double_knock_out_option/double_knock_out_options_01_shard00.json",
        "catalog/product/double_knock_out_option/double_knock_out_options_01_shard00/generation.yaml",
        selected_rows);
}
