// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{6, 19, 23, 34, 47, 53, 65, 78, 85, 94, 103, 119, 120, 138, 149, 157, 165, 173, 182, 190, 204, 219, 221, 231, 246, 255, 268, 278, 281, 297, 303, 315, 326, 334, 340, 354, 361, 375, 383, 395, 401, 419, 427, 437, 441, 455, 466, 475, 482, 490, 500, 513, 527, 531, 541, 556, 567, 579, 583, 591, 600, 610, 620, 630, 646, 655, 662, 679, 686, 691, 701, 710, 729, 739, 748, 750, 765, 775, 783, 796, 805, 811, 822, 834, 844, 852, 860, 873, 880, 891, 903, 911, 921, 930, 948, 951, 961, 977, 984, 999};
    write_product_shard(
        "datasets/product/asian_option/asian_options_01.json",
        "datasets/product/asian_option/asian_options_01_shard00.json",
        "catalog/product/asian_option/asian_options_01_shard00/generation.yaml",
        selected_rows);
}
