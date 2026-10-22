// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{7, 10, 29, 31, 40, 58, 67, 74, 88, 96, 107, 117, 122, 137, 144, 158, 161, 170, 188, 196, 209, 216, 223, 236, 242, 252, 269, 276, 289, 291, 302, 313, 324, 331, 343, 355, 367, 376, 382, 396, 404, 413, 427, 438, 447, 459, 462, 475, 480, 498, 508, 512, 525, 530, 542, 550, 567, 578, 583, 590, 609, 613, 628, 631, 644, 654, 665, 671, 684, 693, 706, 710, 723, 730, 743, 759, 762, 775, 784, 797, 802, 811, 825, 837, 849, 852, 866, 874, 881, 894, 903, 916, 920, 938, 949, 950, 968, 977, 987, 990};
    write_product_shard(
        "datasets/product/digital_option/digital_options_01.json",
        "datasets/product/digital_option/digital_options_01_shard00.json",
        "catalog/product/digital_option/digital_options_01_shard00/generation.yaml",
        selected_rows);
}
