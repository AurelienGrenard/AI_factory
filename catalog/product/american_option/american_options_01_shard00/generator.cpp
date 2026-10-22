// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{0, 14, 21, 33, 49, 59, 69, 76, 85, 96, 109, 116, 126, 133, 145, 159, 162, 179, 181, 194, 201, 210, 222, 239, 249, 252, 263, 278, 285, 290, 309, 317, 328, 330, 343, 353, 362, 370, 380, 396, 409, 419, 427, 439, 441, 459, 467, 479, 487, 496, 502, 513, 525, 536, 545, 559, 565, 577, 589, 594, 603, 617, 626, 634, 648, 650, 660, 679, 683, 696, 701, 716, 725, 738, 740, 754, 768, 777, 788, 793, 803, 817, 825, 839, 842, 858, 865, 876, 884, 898, 907, 919, 925, 930, 948, 958, 967, 975, 983, 993};
    write_product_shard(
        "datasets/product/american_option/american_options_01.json",
        "datasets/product/american_option/american_options_01_shard00.json",
        "catalog/product/american_option/american_options_01_shard00/generation.yaml",
        selected_rows);
}
