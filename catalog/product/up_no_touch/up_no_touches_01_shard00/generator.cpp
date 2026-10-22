// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{4, 11, 22, 39, 45, 54, 61, 77, 88, 91, 108, 114, 124, 131, 140, 155, 167, 175, 187, 192, 200, 219, 229, 238, 245, 256, 268, 276, 285, 295, 306, 314, 327, 339, 348, 355, 366, 378, 380, 397, 409, 415, 429, 435, 442, 457, 469, 475, 482, 494, 502, 516, 522, 533, 548, 552, 569, 571, 586, 598, 603, 610, 622, 633, 645, 652, 666, 673, 685, 693, 709, 716, 720, 732, 744, 750, 761, 772, 781, 792, 809, 813, 825, 834, 841, 855, 866, 871, 888, 891, 903, 918, 921, 934, 949, 955, 964, 973, 989, 990};
    write_product_shard(
        "datasets/product/up_no_touch/up_no_touches_01.json",
        "datasets/product/up_no_touch/up_no_touches_01_shard00.json",
        "catalog/product/up_no_touch/up_no_touches_01_shard00/generation.yaml",
        selected_rows);
}
