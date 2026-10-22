// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{8, 19, 29, 39, 47, 59, 60, 73, 82, 98, 103, 110, 122, 130, 142, 150, 163, 178, 182, 194, 204, 211, 229, 237, 248, 255, 265, 273, 282, 297, 305, 319, 325, 338, 343, 355, 366, 372, 388, 395, 409, 410, 427, 434, 449, 455, 464, 473, 486, 498, 503, 519, 522, 530, 548, 550, 563, 576, 583, 596, 609, 619, 622, 633, 647, 655, 664, 672, 684, 696, 709, 713, 725, 730, 746, 757, 769, 779, 783, 790, 809, 813, 821, 838, 849, 854, 864, 874, 880, 899, 909, 910, 923, 936, 947, 953, 965, 972, 984, 993};
    write_product_shard(
        "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01.json",
        "datasets/product/asset_or_nothing_option/asset_or_nothing_options_01_shard00.json",
        "catalog/product/asset_or_nothing_option/asset_or_nothing_options_01_shard00/generation.yaml",
        selected_rows);
}
