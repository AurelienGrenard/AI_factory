// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{3, 12, 21, 34, 42, 51, 64, 79, 85, 98, 101, 116, 126, 137, 146, 151, 160, 171, 183, 199, 203, 219, 227, 230, 245, 253, 260, 278, 284, 294, 306, 317, 325, 330, 347, 356, 364, 376, 383, 393, 404, 416, 421, 430, 440, 450, 461, 472, 482, 497, 507, 515, 524, 535, 547, 553, 568, 579, 581, 597, 606, 610, 626, 633, 641, 655, 662, 671, 687, 690, 704, 719, 723, 734, 746, 750, 766, 774, 783, 798, 808, 818, 829, 836, 840, 854, 866, 870, 880, 893, 904, 919, 925, 933, 940, 958, 969, 978, 988, 995};
    write_product_shard(
        "datasets/product/athena_autocall/athena_autocalls_01.json",
        "datasets/product/athena_autocall/athena_autocalls_01_shard00.json",
        "catalog/product/athena_autocall/athena_autocalls_01_shard00/generation.yaml",
        selected_rows);
}
