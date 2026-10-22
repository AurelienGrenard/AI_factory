// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{2, 16, 29, 39, 40, 54, 64, 71, 82, 96, 108, 116, 126, 135, 147, 158, 162, 170, 184, 198, 201, 216, 228, 231, 243, 255, 268, 273, 283, 296, 301, 318, 326, 333, 345, 351, 366, 375, 382, 396, 403, 414, 425, 436, 444, 456, 464, 470, 485, 492, 500, 510, 522, 538, 540, 558, 569, 579, 583, 598, 606, 615, 627, 631, 643, 651, 663, 672, 685, 691, 709, 716, 728, 736, 743, 753, 768, 777, 784, 795, 800, 813, 821, 834, 841, 859, 868, 878, 884, 899, 907, 918, 927, 931, 945, 952, 963, 975, 981, 992};
    write_product_shard(
        "datasets/product/down_and_in_option/down_and_in_options_01.json",
        "datasets/product/down_and_in_option/down_and_in_options_01_shard00.json",
        "catalog/product/down_and_in_option/down_and_in_options_01_shard00/generation.yaml",
        selected_rows);
}
