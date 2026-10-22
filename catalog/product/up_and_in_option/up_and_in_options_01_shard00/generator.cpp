// Deterministic product shard, selected from the 1000-row parent dataset.
#include "tools/datasets/product_shard.hpp"
#include <array>
#include <cstddef>
#include <filesystem>

int main() {
    using ai_factory::workbench::datasets::write_product_shard;
    constexpr std::array<std::size_t, 100> selected_rows{6, 11, 22, 39, 48, 57, 68, 76, 81, 97, 102, 118, 127, 134, 142, 152, 167, 178, 181, 198, 207, 217, 227, 237, 242, 252, 260, 274, 280, 292, 306, 317, 325, 336, 341, 354, 368, 375, 386, 396, 404, 413, 428, 433, 448, 458, 469, 472, 483, 494, 507, 513, 521, 538, 540, 559, 561, 573, 589, 595, 608, 617, 627, 637, 641, 654, 666, 678, 688, 693, 703, 710, 720, 737, 747, 754, 765, 775, 783, 791, 803, 812, 823, 838, 841, 854, 867, 875, 886, 896, 905, 918, 927, 932, 944, 959, 969, 977, 988, 993};
    write_product_shard(
        "datasets/product/up_and_in_option/up_and_in_options_01.json",
        "datasets/product/up_and_in_option/up_and_in_options_01_shard00.json",
        "catalog/product/up_and_in_option/up_and_in_options_01_shard00/generation.yaml",
        selected_rows);
}
