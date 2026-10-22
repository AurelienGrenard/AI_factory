// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{1, 12, 21, 37, 41, 56, 65, 74, 87, 91, 106, 114, 125, 131, 146, 153, 163, 172, 188, 191, 203, 214, 229, 236, 242, 259, 260, 277, 284, 291, 305, 315, 329, 337, 345, 357, 365, 371, 385, 397, 401, 417, 422, 437, 443, 454, 461, 475, 481, 494, 507, 517, 525, 536, 548, 559, 561, 573, 582, 594, 608, 618, 629, 634, 641, 659, 663, 677, 685, 696, 707, 711, 726, 738, 745, 754, 760, 778, 780, 798, 802, 815, 826, 831, 843, 858, 863, 879, 883, 892, 908, 912, 927, 933, 944, 955, 966, 975, 980, 998};
    write_product_shard(
        "datasets/product/up_and_out_option/up_and_out_options_01.json",
        "datasets/product/up_and_out_option/up_and_out_options_01_shard00.json",
        "catalog/product/up_and_out_option/up_and_out_options_01_shard00/generation.yaml",
        selected_rows);
}
