// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{2, 12, 28, 32, 44, 56, 68, 77, 81, 96, 102, 115, 128, 139, 142, 157, 161, 171, 183, 192, 202, 214, 220, 238, 242, 254, 266, 274, 280, 293, 301, 316, 321, 338, 343, 355, 365, 372, 381, 395, 400, 415, 423, 431, 440, 455, 464, 477, 488, 499, 508, 512, 522, 535, 541, 551, 564, 575, 585, 591, 604, 614, 621, 633, 641, 652, 661, 674, 688, 696, 700, 713, 728, 739, 746, 759, 768, 776, 783, 798, 808, 813, 826, 838, 849, 850, 864, 876, 888, 897, 907, 910, 926, 936, 948, 958, 968, 970, 985, 992};
    write_product_shard(
        "datasets/product/bermudan_swaption/bermudan_swaptions_01.json",
        "datasets/product/bermudan_swaption/bermudan_swaptions_01_shard00.json",
        "catalog/product/bermudan_swaption/bermudan_swaptions_01_shard00/generation.yaml",
        selected_rows);
}
