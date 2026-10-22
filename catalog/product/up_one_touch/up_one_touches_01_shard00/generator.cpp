// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{3, 16, 21, 34, 40, 52, 69, 75, 87, 99, 108, 113, 120, 138, 147, 151, 169, 173, 186, 195, 200, 217, 222, 234, 245, 257, 261, 273, 289, 291, 304, 311, 322, 336, 347, 358, 361, 376, 385, 393, 404, 419, 424, 439, 449, 453, 469, 475, 485, 493, 503, 515, 521, 533, 546, 553, 562, 576, 581, 599, 608, 611, 628, 631, 646, 653, 667, 673, 684, 697, 709, 716, 724, 731, 748, 753, 763, 777, 788, 790, 807, 819, 825, 834, 844, 858, 860, 874, 887, 897, 908, 915, 928, 938, 942, 955, 964, 979, 988, 992};
    write_product_shard(
        "datasets/product/up_one_touch/up_one_touches_01.json",
        "datasets/product/up_one_touch/up_one_touches_01_shard00.json",
        "catalog/product/up_one_touch/up_one_touches_01_shard00/generation.yaml",
        selected_rows);
}
