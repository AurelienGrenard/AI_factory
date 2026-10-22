// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{8, 17, 20, 39, 40, 53, 63, 75, 83, 97, 109, 119, 127, 138, 140, 157, 165, 177, 183, 190, 200, 211, 227, 231, 241, 258, 267, 273, 284, 296, 307, 312, 326, 338, 340, 355, 360, 378, 385, 390, 404, 414, 424, 433, 449, 450, 464, 472, 484, 499, 508, 519, 520, 534, 543, 551, 565, 579, 584, 597, 609, 616, 627, 631, 648, 650, 665, 678, 682, 694, 705, 716, 727, 730, 747, 755, 765, 778, 781, 793, 805, 814, 822, 839, 843, 853, 862, 879, 888, 899, 904, 915, 926, 934, 948, 950, 963, 972, 982, 992};
    write_product_shard(
        "datasets/product/range_accrual/range_accruals_01.json",
        "datasets/product/range_accrual/range_accruals_01_shard00.json",
        "catalog/product/range_accrual/range_accruals_01_shard00/generation.yaml",
        selected_rows);
}
