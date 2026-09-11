// Ordered CIR-factor parameter construction, shared by CIR and CIR++ recipes.
#pragma once

#include "tools/datasets/sampling.hpp"
#include <cstdint>

namespace ai_factory::workbench::datasets::cir {
// 900 core then 100 stress rows, with conditional volatility spanning Feller regimes.
GeneratedRows generate_core_stress_rows(std::uint64_t seed);
}  // namespace ai_factory::workbench::datasets::cir
