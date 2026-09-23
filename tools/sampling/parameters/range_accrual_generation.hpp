// Reusable host-side construction of equity Range Accrual datasets.
#pragma once

#include "tools/datasets/sampling.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::range_accrual {

// Generate standard terms plus a controlled tail of narrow and wide ranges.
GeneratedRows generate_rows(
    std::size_t core_row_count,
    std::size_t tail_row_count,
    std::uint64_t seed
);

}  // namespace ai_factory::workbench::datasets::range_accrual
