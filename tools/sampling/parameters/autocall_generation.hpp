// Reusable host-side construction of single-underlying autocall datasets.
#pragma once

#include "tools/datasets/sampling.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::autocall {

// Generate Phoenix terms with a separate conditional coupon barrier.
GeneratedRows generate_phoenix_rows(
    std::size_t core_row_count,
    std::size_t tail_row_count,
    std::uint64_t seed
);

// Generate Athena terms whose accrued gain is paid only at redemption.
GeneratedRows generate_athena_rows(
    std::size_t core_row_count,
    std::size_t tail_row_count,
    std::uint64_t seed
);

}  // namespace ai_factory::workbench::datasets::autocall
