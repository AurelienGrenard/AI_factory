// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{7, 13, 29, 35, 42, 52, 61, 79, 82, 90, 104, 111, 125, 131, 144, 151, 166, 171, 181, 192, 204, 217, 225, 236, 249, 252, 268, 274, 282, 291, 306, 310, 321, 333, 348, 355, 368, 374, 388, 394, 405, 416, 423, 438, 445, 457, 461, 476, 486, 491, 500, 512, 525, 538, 545, 555, 562, 575, 580, 596, 603, 611, 625, 639, 648, 651, 666, 672, 685, 698, 706, 711, 720, 736, 743, 750, 761, 778, 787, 794, 806, 815, 820, 836, 842, 852, 860, 870, 880, 891, 907, 910, 926, 931, 945, 959, 964, 976, 982, 990};
    write_product_shard(
        "datasets/product/straddle/straddles_01.json",
        "datasets/product/straddle/straddles_01_shard00.json",
        "catalog/product/straddle/straddles_01_shard00/generation.yaml",
        selected_rows);
}
