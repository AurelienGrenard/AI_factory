// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{5, 19, 23, 35, 47, 57, 61, 70, 88, 94, 103, 119, 125, 132, 146, 155, 163, 174, 189, 192, 209, 214, 225, 238, 244, 259, 267, 270, 281, 297, 300, 318, 328, 336, 349, 359, 363, 371, 387, 392, 407, 414, 421, 431, 443, 453, 465, 473, 484, 494, 501, 513, 523, 534, 545, 555, 562, 578, 586, 599, 602, 611, 620, 635, 644, 654, 663, 673, 688, 690, 704, 713, 727, 734, 742, 755, 766, 776, 787, 799, 801, 814, 824, 838, 844, 858, 862, 876, 884, 892, 904, 916, 921, 930, 947, 950, 965, 971, 989, 999};
    write_product_shard(
        "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01.json",
        "datasets/product/phoenix_memory_autocall/phoenix_memory_autocalls_01_shard00.json",
        "catalog/product/phoenix_memory_autocall/phoenix_memory_autocalls_01_shard00/generation.yaml",
        selected_rows);
}
