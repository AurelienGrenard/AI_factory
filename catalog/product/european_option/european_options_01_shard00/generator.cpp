// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{2, 15, 22, 30, 43, 50, 67, 76, 85, 96, 107, 118, 121, 138, 148, 156, 169, 171, 182, 195, 200, 215, 226, 239, 247, 257, 264, 271, 280, 292, 304, 317, 328, 333, 347, 352, 368, 374, 389, 397, 407, 414, 429, 433, 446, 456, 468, 475, 486, 494, 504, 519, 529, 530, 540, 559, 568, 572, 587, 592, 603, 613, 629, 639, 643, 659, 669, 672, 683, 695, 706, 712, 727, 736, 741, 758, 764, 771, 782, 792, 806, 814, 820, 836, 847, 852, 869, 871, 884, 897, 902, 917, 926, 935, 948, 957, 966, 974, 986, 992};
    write_product_shard(
        "datasets/product/european_option/european_options_01.json",
        "datasets/product/european_option/european_options_01_shard00.json",
        "catalog/product/european_option/european_options_01_shard00/generation.yaml",
        selected_rows);
}
