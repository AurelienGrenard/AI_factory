// Reusable host-side construction of single-underlying Cliquet datasets.
#pragma once

#include "tools/datasets/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::datasets::cliquet {

// Generate locally and globally capped periodic-return Cliquet terms.
GeneratedRows generate_rows(
    std::size_t core_row_count,
    std::size_t tail_row_count,
    std::uint64_t seed
);

}  // namespace ai_factory::workbench::datasets::cliquet
