// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{5, 14, 25, 36, 44, 51, 67, 74, 83, 91, 105, 114, 126, 135, 141, 156, 162, 172, 185, 191, 206, 213, 228, 238, 242, 254, 261, 273, 280, 296, 303, 311, 329, 334, 347, 357, 363, 376, 385, 397, 408, 411, 424, 438, 440, 458, 469, 478, 482, 497, 505, 513, 524, 532, 547, 556, 569, 578, 584, 594, 600, 612, 628, 631, 649, 653, 662, 674, 682, 696, 704, 716, 727, 732, 743, 757, 765, 774, 785, 792, 805, 817, 823, 837, 841, 856, 867, 879, 889, 896, 906, 917, 921, 934, 945, 950, 968, 971, 981, 995};
    write_product_shard(
        "datasets/product/down_and_out_option/down_and_out_options_01.json",
        "datasets/product/down_and_out_option/down_and_out_options_01_shard00.json",
        "catalog/product/down_and_out_option/down_and_out_options_01_shard00/generation.yaml",
        selected_rows);
}
