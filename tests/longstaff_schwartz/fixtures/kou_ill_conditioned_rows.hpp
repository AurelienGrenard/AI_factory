// Frozen catalogue rows sensitive to LSM Gram rounding; CPU binary128 regression reference.
// Seeds preserve each original row offset. This is a regression fixture, not price certification.
#pragma once

#include "model/equity/markovian/kou/parameters.hpp"
#include "product/american_option/parameters.hpp"
#include <array>
#include <cstdint>

namespace ai_factory::tests::longstaff_schwartz {
struct KouRegressionCase {
    workbench::model::equity::kou::ModelParameters model;
    workbench::product::AmericanOptionParameters product;
    std::uint64_t seed;
    float put_price;
    float put_standard_error;
};

inline constexpr std::array<KouRegressionCase, 8> kKouRegressionCases{{
    { // Catalogue row 000230
        {1.0f, 0.0770503953f, 0.0596261695f, 0.43655473f,
         0.484757572f, 0.426682383f, 4.92140961f, 19.7259369f},
        {0.991473496f, 205U, 10U},
        11668827270172115173ULL, 0.146392301f, 0.000152857843f
    },
    { // Catalogue row 000271
        {1.0f, 0.0531867258f, 0.0401320867f, 0.413183808f,
         0.741986752f, 0.399550527f, 9.02534294f, 17.385252f},
        {1.00999117f, 238U, 5U},
        11668827270172115214ULL, 0.159000978f, 0.000159251824f
    },
    { // Catalogue row 000422
        {1.0f, 0.0534107946f, 0.0240393691f, 0.272807211f,
         0.944705009f, 0.616330862f, 17.4004936f, 11.3345442f},
        {0.767850339f, 372U, 5U},
        11668827270172115365ULL, 0.0314226002f, 6.27464615e-05f
    },
    { // Catalogue row 000693
        {1.0f, 0.0042944476f, 0.0165097006f, 0.40864718f,
         0.820132196f, 0.54718107f, 15.2720184f, 15.9308348f},
        {1.13090229f, 589U, 5U},
        11668827270172115636ULL, 0.345449746f, 0.000286294904f
    },
    { // Catalogue row 000719
        {1.0f, 0.0505662188f, 0.00381520297f, 0.386917651f,
         0.402786344f, 0.281548917f, 12.3898716f, 11.8696022f},
        {1.53775847f, 606U, 5U},
        11668827270172115662ULL, 0.568938732f, 0.000246145064f
    },
    { // Catalogue row 000796
        {1.0f, 0.0160987489f, 0.00107208418f, 0.297572702f,
         0.718412578f, 0.398579478f, 11.6897259f, 19.1306496f},
        {1.36175179f, 672U, 5U},
        11668827270172115739ULL, 0.427751333f, 0.00024789662f
    },
    { // Catalogue row 000807
        {1.0f, 0.0495738164f, 0.0399957672f, 0.230944753f,
         0.484055668f, 0.582063258f, 7.09555864f, 12.093235f},
        {0.817534626f, 689U, 5U},
        11668827270172115750ULL, 0.0653239712f, 9.50188623e-05f
    },
    { // Catalogue row 000873
        {1.0f, 0.02486117f, 0.0151298651f, 0.189267665f,
         0.316085815f, 0.467371404f, 6.15455723f, 15.3124294f},
        {1.16689241f, 739U, 5U},
        11668827270172115816ULL, 0.231144726f, 0.000166725891f
    },
}};
}  // namespace ai_factory::tests::longstaff_schwartz
