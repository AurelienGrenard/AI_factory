// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{0, 12, 29, 32, 42, 57, 60, 74, 84, 98, 105, 113, 128, 131, 148, 152, 169, 176, 183, 196, 200, 213, 227, 232, 249, 252, 264, 276, 283, 299, 304, 313, 329, 336, 344, 351, 361, 372, 381, 391, 402, 418, 428, 430, 443, 456, 466, 475, 488, 498, 505, 511, 523, 536, 548, 559, 564, 573, 585, 592, 602, 618, 625, 632, 647, 656, 667, 673, 689, 698, 701, 717, 724, 739, 748, 757, 768, 777, 784, 799, 806, 810, 821, 833, 843, 856, 863, 879, 889, 899, 901, 915, 920, 931, 946, 954, 961, 978, 980, 990};
    write_product_shard(
        "datasets/product/geometric_asian_option/geometric_asian_options_01.json",
        "datasets/product/geometric_asian_option/geometric_asian_options_01_shard00.json",
        "catalog/product/geometric_asian_option/geometric_asian_options_01_shard00/generation.yaml",
        selected_rows);
}
