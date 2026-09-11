// Frozen NUM-021 catalogue rows; decimal literals round back to the source FP32.
// Keep these inputs independent of locally generated, ignored JSON datasets.
#pragma once

#include "model/fixed_income/cir/parameters.hpp"
#include "model/fixed_income/vasicek/parameters.hpp"
#include "product/european_swaption/parameters.hpp"
#include <array>

namespace jamshidian_stress_fixtures {

template<class Model>
struct Row {
    unsigned int source_id;
    Model model;
    ai_factory::workbench::product::RegularEuropeanSwaptionParameters product;
};

inline constexpr std::array<Row<ai_factory::workbench::model::fixed_income::cir::ModelParameters>, 8> cir_rows = {{
    {910U, {{2.078552484512329f, 0.1049085482954979f, 0.571814239025116f}, 0.12824992835521698f},
        {1.0f, 0.0010000000474974513f, 0.0833333358168602f, 1U, 21U, 600U}},
    {917U, {{2.3375487327575684f, 0.14969220757484436f, 0.501526951789856f}, 0.009378142654895782f},
        {1.0f, 0.0f, 1.0f, 1U, 252U, 50U}},
    {930U, {{0.4356035590171814f, 0.1976194679737091f, 0.2930179238319397f}, 0.09142613410949707f},
        {1.0f, 0.0010000000474974513f, 0.0833333358168602f, 5U, 21U, 600U}},
    {949U, {{1.3311567306518555f, 0.17828887701034546f, 0.2767195999622345f}, 0.07589762657880783f},
        {1.0f, 0.0f, 0.0833333358168602f, 21U, 21U, 600U}},
    {953U, {{2.1520187854766846f, 0.19518226385116577f, 0.31367743015289307f}, 0.026100996881723404f},
        {1.0f, 0.0f, 0.25f, 21U, 63U, 200U}},
    {958U, {{0.25887054204940796f, 0.1444702446460724f, 0.12965041399002075f}, 0.16375575959682465f},
        {1.0f, 0.0010000000474974513f, 1.0f, 21U, 252U, 50U}},
    {989U, {{0.706378698348999f, 0.15055815875530243f, 0.13484551012516022f}, 0.18681620061397552f},
        {1.0f, 0.0f, 0.0833333358168602f, 12600U, 21U, 600U}},
    {993U, {{0.7221033573150635f, 0.1919848769903183f, 0.25034841895103455f}, 0.15112964808940887f},
        {1.0f, 0.0f, 0.25f, 12600U, 63U, 200U}},
}};

inline constexpr std::array<Row<ai_factory::workbench::model::fixed_income::vasicek::ModelParameters>, 4> vasicek_rows = {{
    {950U, {{0.6310370564460754f, 0.022618891671299934f, 0.01836942695081234f}, 0.10835930705070496f},
        {1.0f, 0.0010000000474974513f, 0.0833333358168602f, 21U, 21U, 600U}},
    {951U, {{0.009088853374123573f, 0.062236517667770386f, 0.0022070237901061773f}, 0.0971875935792923f},
        {1.0f, 0.15000000596046448f, 0.0833333358168602f, 21U, 21U, 600U}},
    {974U, {{0.0580260306596756f, 0.09566614031791687f, 0.000669299450237304f}, 0.006982237100601196f},
        {1.0f, 0.0010000000474974513f, 0.25f, 7560U, 63U, 200U}},
    {997U, {{1.069475769996643f, 0.1271408647298813f, 0.04266421124339104f}, -0.04122953861951828f},
        {1.0f, 0.0f, 1.0f, 12600U, 252U, 50U}},
}};

}  // namespace jamshidian_stress_fixtures
