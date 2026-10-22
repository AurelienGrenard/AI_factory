// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{0, 18, 27, 31, 47, 52, 65, 76, 88, 95, 108, 117, 121, 136, 147, 154, 167, 171, 183, 198, 203, 210, 222, 231, 246, 252, 269, 274, 285, 290, 307, 311, 326, 338, 347, 357, 369, 374, 381, 396, 405, 412, 420, 437, 445, 459, 468, 474, 484, 495, 509, 517, 527, 536, 543, 555, 565, 578, 583, 599, 602, 614, 626, 633, 647, 658, 666, 676, 682, 698, 704, 712, 721, 731, 744, 757, 767, 778, 789, 791, 809, 816, 824, 837, 840, 854, 861, 873, 884, 896, 900, 910, 923, 937, 946, 954, 964, 976, 980, 994};
    write_product_shard(
        "datasets/product/rate_option/rate_options_01.json",
        "datasets/product/rate_option/rate_options_01_shard00.json",
        "catalog/product/rate_option/rate_options_01_shard00/generation.yaml",
        selected_rows);
}
