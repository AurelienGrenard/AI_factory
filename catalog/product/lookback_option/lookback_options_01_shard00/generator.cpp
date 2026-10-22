// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{1, 17, 27, 36, 45, 57, 60, 70, 84, 91, 101, 118, 122, 131, 146, 157, 160, 174, 187, 193, 203, 212, 229, 237, 241, 250, 261, 273, 289, 295, 300, 318, 328, 331, 341, 357, 362, 375, 381, 395, 401, 418, 421, 433, 445, 452, 462, 473, 489, 493, 507, 512, 521, 539, 544, 555, 564, 572, 581, 597, 600, 617, 622, 632, 644, 651, 664, 672, 685, 691, 706, 713, 727, 731, 741, 756, 765, 774, 789, 796, 800, 812, 824, 833, 842, 858, 866, 873, 888, 896, 905, 918, 923, 934, 949, 950, 961, 973, 982, 994};
    write_product_shard(
        "datasets/product/lookback_option/lookback_options_01.json",
        "datasets/product/lookback_option/lookback_options_01_shard00.json",
        "catalog/product/lookback_option/lookback_options_01_shard00/generation.yaml",
        selected_rows);
}
