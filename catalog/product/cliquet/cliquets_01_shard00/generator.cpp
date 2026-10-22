// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{1, 16, 23, 39, 45, 52, 67, 73, 84, 91, 102, 110, 121, 135, 148, 152, 164, 177, 188, 190, 200, 214, 220, 231, 248, 256, 267, 275, 284, 297, 308, 311, 326, 331, 345, 356, 368, 378, 382, 394, 408, 414, 427, 439, 443, 459, 469, 474, 481, 495, 502, 513, 523, 536, 546, 557, 566, 572, 581, 592, 609, 613, 625, 639, 646, 651, 660, 675, 681, 695, 707, 718, 727, 739, 741, 756, 762, 777, 789, 798, 802, 817, 824, 833, 843, 854, 866, 870, 883, 897, 904, 919, 920, 931, 940, 954, 968, 977, 988, 993};
    write_product_shard(
        "datasets/product/cliquet/cliquets_01.json",
        "datasets/product/cliquet/cliquets_01_shard00.json",
        "catalog/product/cliquet/cliquets_01_shard00/generation.yaml",
        selected_rows);
}
