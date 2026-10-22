// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{3, 13, 21, 36, 45, 50, 68, 72, 80, 93, 109, 114, 129, 137, 143, 153, 164, 174, 182, 199, 209, 212, 221, 237, 242, 257, 268, 276, 283, 298, 307, 318, 321, 336, 343, 357, 364, 370, 386, 399, 405, 415, 427, 437, 448, 453, 463, 472, 483, 495, 508, 513, 526, 535, 542, 551, 567, 571, 581, 599, 605, 618, 628, 637, 649, 656, 664, 675, 688, 693, 705, 714, 727, 730, 748, 755, 763, 770, 784, 796, 807, 813, 823, 836, 841, 850, 867, 873, 885, 899, 902, 915, 929, 934, 945, 958, 961, 973, 982, 991};
    write_product_shard(
        "datasets/product/gap_option/gap_call_options_01.json",
        "datasets/product/gap_option/gap_call_options_01_shard00.json",
        "catalog/product/gap_option/gap_call_options_01_shard00/generation.yaml",
        selected_rows);
}
