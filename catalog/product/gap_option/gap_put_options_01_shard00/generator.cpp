// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{7, 15, 21, 32, 48, 54, 67, 75, 80, 98, 106, 110, 129, 133, 146, 159, 162, 176, 181, 190, 202, 216, 225, 234, 244, 253, 268, 274, 286, 293, 307, 312, 322, 335, 342, 353, 363, 373, 381, 398, 409, 417, 421, 434, 447, 455, 460, 476, 487, 493, 505, 517, 523, 530, 540, 551, 567, 579, 588, 591, 603, 610, 626, 631, 649, 656, 668, 672, 686, 695, 709, 712, 724, 735, 745, 753, 763, 775, 781, 798, 804, 812, 829, 838, 843, 859, 861, 876, 886, 898, 905, 919, 925, 931, 941, 952, 964, 970, 980, 998};
    write_product_shard(
        "datasets/product/gap_option/gap_put_options_01.json",
        "datasets/product/gap_option/gap_put_options_01_shard00.json",
        "catalog/product/gap_option/gap_put_options_01_shard00/generation.yaml",
        selected_rows);
}
