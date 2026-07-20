#pragma once
#include "ai_factory/cuda/common/types.cuh"
#include <cstddef>
namespace ai_factory::cpu::g2_plus_plus {void price_swaption_batch(const cuda::G2PlusPlusSwaptionRow*,std::size_t,std::size_t,cuda::MonteCarloOutput*);}
