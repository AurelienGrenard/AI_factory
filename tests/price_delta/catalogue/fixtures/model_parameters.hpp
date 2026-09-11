// Compact non-degenerate model parameters for bounded public catalogue checks.
#pragma once

#include "model/equity/markovian/black_scholes/parameters.hpp"
#include "model/equity/markovian/bates/parameters.hpp"
#include "model/equity/markovian/cev/parameters.hpp"
#include "model/equity/markovian/heston/parameters.hpp"
#include "model/equity/markovian/heston_3_2/parameters.hpp"
#include "model/equity/markovian/kou/parameters.hpp"
#include "model/equity/markovian/merton/parameters.hpp"
#include "model/equity/markovian/normal_inverse_gaussian/parameters.hpp"
#include "model/equity/markovian/sabr/parameters.hpp"
#include "model/equity/markovian/schobel_zhu/parameters.hpp"
#include "model/equity/markovian/stein_stein/parameters.hpp"
#include "model/equity/markovian/variance_gamma/parameters.hpp"

namespace price_delta_test::fixtures {
inline constexpr ai_factory::workbench::model::equity::black_scholes::ModelParameters black_scholes{1.2f,.03f,.01f,.2f};
inline constexpr ai_factory::workbench::model::equity::bates::ModelParameters bates{1.2f,.03f,.01f,.04f,1.5f,.04f,.4f,-.7f,.4f,-.05f,.12f};
inline constexpr ai_factory::workbench::model::equity::cev::ModelParameters cev{1.2f,.03f,.01f,.3f,.6f};
inline constexpr ai_factory::workbench::model::equity::heston::ModelParameters heston{1.2f,.03f,.01f,.04f,1.5f,.04f,.4f,-.7f};
inline constexpr ai_factory::workbench::model::equity::heston_3_2::ModelParameters heston_3_2{1.2f,.03f,.01f,.04f,1.5f,.04f,.4f,-.7f};
inline constexpr ai_factory::workbench::model::equity::kou::ModelParameters kou{1.2f,.03f,.01f,.2f,.4f,.3f,12.f,8.f};
inline constexpr ai_factory::workbench::model::equity::merton::ModelParameters merton{1.2f,.03f,.01f,.2f,.4f,-.05f,.12f};
inline constexpr ai_factory::workbench::model::equity::normal_inverse_gaussian::ModelParameters normal_inverse_gaussian{1.2f,.03f,.01f,15.f,-3.f,.3f};
inline constexpr ai_factory::workbench::model::equity::sabr::ModelParameters sabr{1.2f,.03f,.01f,.2f,.4f,-.5f,.6f};
inline constexpr ai_factory::workbench::model::equity::schobel_zhu::ModelParameters schobel_zhu{1.2f,.03f,.01f,.2f,1.5f,.2f,.3f,-.5f};
inline constexpr ai_factory::workbench::model::equity::stein_stein::ModelParameters stein_stein{1.2f,.03f,.01f,.2f,1.5f,.3f,-.5f};
inline constexpr ai_factory::workbench::model::equity::variance_gamma::ModelParameters variance_gamma{1.2f,.03f,.01f,.2f,.15f,-.1f};
}  // namespace price_delta_test::fixtures
