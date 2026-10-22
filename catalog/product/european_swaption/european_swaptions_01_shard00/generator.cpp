// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{0, 18, 20, 30, 49, 53, 69, 74, 88, 92, 105, 119, 128, 133, 143, 158, 169, 172, 184, 196, 206, 213, 226, 238, 247, 258, 269, 271, 282, 295, 301, 311, 323, 333, 348, 356, 365, 373, 388, 396, 406, 416, 429, 432, 447, 453, 460, 472, 486, 492, 502, 510, 524, 538, 547, 553, 567, 573, 583, 597, 604, 617, 621, 630, 640, 658, 660, 678, 682, 690, 707, 711, 726, 739, 740, 758, 769, 776, 783, 791, 809, 810, 827, 830, 840, 858, 866, 875, 887, 898, 907, 919, 928, 932, 942, 953, 967, 977, 985, 992};
    write_product_shard(
        "datasets/product/european_swaption/european_swaptions_01.json",
        "datasets/product/european_swaption/european_swaptions_01_shard00.json",
        "catalog/product/european_swaption/european_swaptions_01_shard00/generation.yaml",
        selected_rows);
}
