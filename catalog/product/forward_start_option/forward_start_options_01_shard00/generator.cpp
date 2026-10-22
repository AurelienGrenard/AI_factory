// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{1, 16, 27, 32, 40, 50, 64, 79, 81, 94, 103, 110, 123, 132, 148, 159, 167, 177, 187, 194, 202, 218, 220, 239, 242, 252, 264, 271, 282, 295, 307, 317, 320, 338, 345, 351, 365, 374, 389, 391, 406, 411, 424, 431, 443, 452, 461, 471, 485, 492, 505, 512, 529, 535, 549, 556, 563, 570, 583, 594, 601, 616, 621, 636, 647, 650, 669, 673, 689, 691, 709, 717, 723, 739, 743, 754, 767, 776, 788, 799, 800, 815, 822, 832, 848, 853, 862, 873, 887, 890, 909, 915, 925, 933, 945, 952, 969, 971, 981, 999};
    write_product_shard(
        "datasets/product/forward_start_option/forward_start_options_01.json",
        "datasets/product/forward_start_option/forward_start_options_01_shard00.json",
        "catalog/product/forward_start_option/forward_start_options_01_shard00/generation.yaml",
        selected_rows);
}
