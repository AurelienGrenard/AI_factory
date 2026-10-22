// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{5, 19, 24, 36, 44, 53, 69, 74, 86, 97, 105, 116, 126, 132, 145, 157, 162, 176, 180, 194, 200, 214, 221, 236, 243, 259, 263, 279, 284, 298, 307, 313, 324, 334, 340, 354, 368, 379, 384, 390, 409, 411, 427, 433, 440, 452, 462, 472, 488, 493, 506, 514, 529, 537, 545, 553, 563, 578, 586, 590, 606, 614, 626, 630, 648, 653, 665, 676, 681, 691, 704, 711, 720, 738, 746, 751, 768, 775, 783, 799, 805, 810, 827, 839, 848, 857, 860, 873, 888, 899, 904, 913, 923, 939, 940, 959, 962, 970, 989, 991};
    write_product_shard(
        "datasets/product/phoenix_autocall/phoenix_autocalls_01.json",
        "datasets/product/phoenix_autocall/phoenix_autocalls_01_shard00.json",
        "catalog/product/phoenix_autocall/phoenix_autocalls_01_shard00/generation.yaml",
        selected_rows);
}
