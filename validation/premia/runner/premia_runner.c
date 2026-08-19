/* Uniform batch adapter around the stateful Premia C API. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "error_msg.h"
#include "premia_obj.h"
#include "tools.h"
#include "var.h"

#define INPUT_LINE_CAPACITY 4096
#define MAX_NUMERIC_FIELDS 20
#define PREMIA_MC_ITERATIONS 65536L
#define PREMIA_PATH_MC_ITERATIONS 4096L

typedef enum ModelKind {
    MODEL_BLACK_SCHOLES,
    MODEL_HESTON,
    MODEL_BATES,
    MODEL_MERTON,
    MODEL_KOU,
    MODEL_VARIANCE_GAMMA,
    MODEL_NORMAL_INVERSE_GAUSSIAN,
    MODEL_CEV,
    MODEL_STEIN,
    MODEL_VASICEK,
    MODEL_ORNSTEIN_UHLENBECK,
    MODEL_HULL_WHITE,
    MODEL_G2_PLUS_PLUS
} ModelKind;


typedef enum ContractKind {
    CONTRACT_EUROPEAN_CALL,
    CONTRACT_EUROPEAN_PUT,
    CONTRACT_DIGITAL_CALL,
    CONTRACT_SINGLE_BARRIER,
    CONTRACT_SINGLE_BARRIER_REBATE,
    CONTRACT_DOUBLE_BARRIER,
    CONTRACT_FIXED_LOOKBACK_CALL,
    CONTRACT_FIXED_ASIAN,
    CONTRACT_AMERICAN_CALL,
    CONTRACT_AMERICAN_PUT,
    CONTRACT_ZERO_COUPON_CALL,
    CONTRACT_ZERO_COUPON_PUT,
    CONTRACT_CAP,
    CONTRACT_FLOOR
} ContractKind;

typedef struct ModeSpec {
    const char *mode;
    ModelKind model_kind;
    ContractKind contract_kind;
    const char *asset_name;
    const char *model_name;
    const char *option_name;
    const char *method_name;
    int field_count;
} ModeSpec;

static const ModeSpec MODE_SPECS[] = {
    {"black_scholes_european_call", MODEL_BLACK_SCHOLES,
     CONTRACT_EUROPEAN_CALL, "equity_Black_Scholes_type",
     "BlackScholes1dim", "CallEuro", "CF_Call", 6},
    {"black_scholes_european_put", MODEL_BLACK_SCHOLES,
     CONTRACT_EUROPEAN_PUT, "equity_Black_Scholes_type",
     "BlackScholes1dim", "PutEuro", "CF_Put", 6},
    {"black_scholes_digital_call", MODEL_BLACK_SCHOLES,
     CONTRACT_DIGITAL_CALL, "equity_Black_Scholes_type",
     "BlackScholes1dim", "DigitEuro", "CF_Digit", 6},
    {"black_scholes_up_and_out_call", MODEL_BLACK_SCHOLES,
     CONTRACT_SINGLE_BARRIER, "equity_Black_Scholes_type",
     "BlackScholes1dim", "CallUpOutEuro", "CF_CallUpOut", 7},
    {"black_scholes_up_and_in_call", MODEL_BLACK_SCHOLES,
     CONTRACT_SINGLE_BARRIER, "equity_Black_Scholes_type",
     "BlackScholes1dim", "CallUpInEuro", "CF_CallUpIn", 7},
    {"black_scholes_down_and_out_put", MODEL_BLACK_SCHOLES,
     CONTRACT_SINGLE_BARRIER, "equity_Black_Scholes_type",
     "BlackScholes1dim", "PutDownOutEuro", "CF_PutDownOut", 7},
    {"black_scholes_down_and_in_put", MODEL_BLACK_SCHOLES,
     CONTRACT_SINGLE_BARRIER, "equity_Black_Scholes_type",
     "BlackScholes1dim", "PutDownInEuro", "CF_PutDownIn", 7},
    {"black_scholes_double_knock_out_call", MODEL_BLACK_SCHOLES,
     CONTRACT_DOUBLE_BARRIER, "equity_Black_Scholes_type",
     "BlackScholes1dim", "DoubleCallOutEuro",
     "CF_CallOut_KunitomoIkeda", 8},
    {"black_scholes_double_knock_out_put", MODEL_BLACK_SCHOLES,
     CONTRACT_DOUBLE_BARRIER, "equity_Black_Scholes_type",
     "BlackScholes1dim", "DoublePutOutEuro",
     "CF_PutOut_KunitomoIkeda", 8},
    {"black_scholes_lookback_option", MODEL_BLACK_SCHOLES,
     CONTRACT_FIXED_LOOKBACK_CALL, "equity_Black_Scholes_type",
     "BlackScholes1dim", "LookBackCallFixedEuro",
     "CF_Fixed_CallLookBack", 6},
    {"black_scholes_asian_call", MODEL_BLACK_SCHOLES,
     CONTRACT_FIXED_ASIAN, "equity_Black_Scholes_type",
     "BlackScholes1dim", "AsianCallFixedEuro",
     "MC_FixedAsian_ExactMethod", 6},
    {"black_scholes_asian_put", MODEL_BLACK_SCHOLES,
     CONTRACT_FIXED_ASIAN, "equity_Black_Scholes_type",
     "BlackScholes1dim", "AsianPutFixedEuro",
     "MC_FixedAsian_ExactMethod", 6},
    {"black_scholes_up_in_call_rebate", MODEL_BLACK_SCHOLES,
     CONTRACT_SINGLE_BARRIER_REBATE, "equity_Black_Scholes_type",
     "BlackScholes1dim", "CallUpInEuro", "CF_CallUpIn", 8},
    {"heston_european_call", MODEL_HESTON, CONTRACT_EUROPEAN_CALL,
     "equity_stochastic_volatility", "Heston1dim", "CallEuro",
     "CF_Call_Heston", 10},
    {"heston_european_put", MODEL_HESTON, CONTRACT_EUROPEAN_PUT,
     "equity_stochastic_volatility", "Heston1dim", "PutEuro",
     "CF_Put_Heston", 10},
    {"heston_up_and_out_call", MODEL_HESTON, CONTRACT_SINGLE_BARRIER,
     "equity_stochastic_volatility", "Heston1dim", "CallUpOutEuro",
     "AP_FastWHBar_HES", 11},
    {"heston_down_and_out_put", MODEL_HESTON, CONTRACT_SINGLE_BARRIER,
     "equity_stochastic_volatility", "Heston1dim", "PutDownOutEuro",
     "AP_FastWHBar_HES", 11},
    {"heston_asian_call", MODEL_HESTON, CONTRACT_FIXED_ASIAN,
     "equity_stochastic_volatility", "Heston1dim", "AsianCallFixedEuro",
     "AP_FJM_ASIAN_HESTON", 10},
    {"heston_american_call", MODEL_HESTON, CONTRACT_AMERICAN_CALL,
     "equity_stochastic_volatility", "Heston1dim", "CallAmer",
     "AP_FastWHAmer_HES", 10},
    {"heston_american_put", MODEL_HESTON, CONTRACT_AMERICAN_PUT,
     "equity_stochastic_volatility", "Heston1dim", "PutAmer",
     "AP_FastWHAmer_HES", 10},
    {"bates_european_call", MODEL_BATES, CONTRACT_EUROPEAN_CALL,
     "equity_stochastic_volatility", "MertonHeston1dim", "CallEuro",
     "CF_Call_MerHes", 13},
    {"bates_european_put", MODEL_BATES, CONTRACT_EUROPEAN_PUT,
     "equity_stochastic_volatility", "MertonHeston1dim", "PutEuro",
     "CF_Put_MerHes", 13},
    {"bates_up_and_out_call", MODEL_BATES, CONTRACT_SINGLE_BARRIER,
     "equity_stochastic_volatility", "MertonHeston1dim", "CallUpOutEuro",
     "MC_Alfonsi_Bates_Out", 14},
    {"bates_down_and_out_put", MODEL_BATES, CONTRACT_SINGLE_BARRIER,
     "equity_stochastic_volatility", "MertonHeston1dim", "PutDownOutEuro",
     "MC_Alfonsi_Bates_Out", 14},
    {"bates_asian_call", MODEL_BATES, CONTRACT_FIXED_ASIAN,
     "equity_stochastic_volatility", "MertonHeston1dim",
     "AsianCallFixedEuro", "MC_Alfonsi_Asian_Bates", 13},
    {"bates_asian_put", MODEL_BATES, CONTRACT_FIXED_ASIAN,
     "equity_stochastic_volatility", "MertonHeston1dim",
     "AsianPutFixedEuro", "MC_Alfonsi_Asian_Bates", 13},
    {"bates_american_call", MODEL_BATES, CONTRACT_AMERICAN_CALL,
     "equity_stochastic_volatility", "MertonHeston1dim", "CallAmer",
     "MC_AM_Alfonsi_LongstaffSchwartz_Bates", 13},
    {"bates_american_put", MODEL_BATES, CONTRACT_AMERICAN_PUT,
     "equity_stochastic_volatility", "MertonHeston1dim", "PutAmer",
     "MC_AM_Alfonsi_LongstaffSchwartz_Bates", 13},
    {"merton_european_call", MODEL_MERTON, CONTRACT_EUROPEAN_CALL,
     "equity_with_jumps", "Merton1dim", "CallEuro", "CF_Call_Merton", 9},
    {"merton_european_put", MODEL_MERTON, CONTRACT_EUROPEAN_PUT,
     "equity_with_jumps", "Merton1dim", "PutEuro", "CF_Put_Merton", 9},
    {"merton_digital_call", MODEL_MERTON, CONTRACT_DIGITAL_CALL,
     "equity_with_jumps", "Merton1dim", "DigitEuro", "MC_Merton", 9},
    {"merton_up_and_out_call", MODEL_MERTON, CONTRACT_SINGLE_BARRIER,
     "equity_with_jumps", "Merton1dim", "CallUpOutEuro",
     "FD_ImpExpUpOut", 10},
    {"merton_down_and_out_put", MODEL_MERTON, CONTRACT_SINGLE_BARRIER,
     "equity_with_jumps", "Merton1dim", "PutDownOutEuro",
     "FD_ImpExpDownOut", 10},
    {"merton_asian_call", MODEL_MERTON, CONTRACT_FIXED_ASIAN,
     "equity_with_jumps", "Merton1dim", "AsianCallFixedEuro",
     "AP_Asian_FMM_Mer", 9},
    {"merton_asian_put", MODEL_MERTON, CONTRACT_FIXED_ASIAN,
     "equity_with_jumps", "Merton1dim", "AsianPutFixedEuro",
     "AP_Asian_FMM_Mer", 9},
    {"merton_lookback_option", MODEL_MERTON, CONTRACT_FIXED_LOOKBACK_CALL,
     "equity_with_jumps", "Merton1dim", "LookBackCallFixedEuro",
     "MC_Merton_FixedLookback", 9},
    {"kou_european_call", MODEL_KOU, CONTRACT_EUROPEAN_CALL,
     "equity_with_jumps", "Kou1dim", "CallEuro", "AP_Carr_Kou", 10},
    {"kou_european_put", MODEL_KOU, CONTRACT_EUROPEAN_PUT,
     "equity_with_jumps", "Kou1dim", "PutEuro", "AP_Carr_Kou", 10},
    {"kou_digital_call", MODEL_KOU, CONTRACT_DIGITAL_CALL,
     "equity_with_jumps", "Kou1dim", "DigitEuro",
     "AP_Kou_Eu", 10},
    {"kou_up_and_out_call", MODEL_KOU, CONTRACT_SINGLE_BARRIER,
     "equity_with_jumps", "Kou1dim", "CallUpOutEuro",
     "AP_Kou_Barrier_Out", 11},
    {"kou_up_and_in_call", MODEL_KOU, CONTRACT_SINGLE_BARRIER,
     "equity_with_jumps", "Kou1dim", "CallUpInEuro",
     "AP_Kou_Barrier_In", 11},
    {"kou_down_and_out_put", MODEL_KOU, CONTRACT_SINGLE_BARRIER,
     "equity_with_jumps", "Kou1dim", "PutDownOutEuro",
     "AP_Kou_Barrier_Out", 11},
    {"kou_down_and_in_put", MODEL_KOU, CONTRACT_SINGLE_BARRIER,
     "equity_with_jumps", "Kou1dim", "PutDownInEuro",
     "AP_Kou_Barrier_In", 11},
    {"kou_asian_call", MODEL_KOU, CONTRACT_FIXED_ASIAN,
     "equity_with_jumps", "Kou1dim", "AsianCallFixedEuro",
     "AP_Asian_FMM_KOU", 11},
    {"kou_asian_put", MODEL_KOU, CONTRACT_FIXED_ASIAN,
     "equity_with_jumps", "Kou1dim", "AsianPutFixedEuro",
     "AP_Asian_FMM_KOU", 11},
    {"kou_lookback_option", MODEL_KOU, CONTRACT_FIXED_LOOKBACK_CALL,
     "equity_with_jumps", "Kou1dim", "LookBackCallFixedEuro",
     "AP_Kou_LookbackFixed", 10},
    {"kou_up_in_call_rebate", MODEL_KOU,
     CONTRACT_SINGLE_BARRIER_REBATE, "equity_with_jumps", "Kou1dim",
     "CallUpInEuro", "AP_Kou_Barrier_In", 12},
    {"variance_gamma_european_call", MODEL_VARIANCE_GAMMA,
     CONTRACT_EUROPEAN_CALL, "equity_with_jumps", "VarianceGamma1dim",
     "CallEuro", "AP_Carr_VG", 8},
    {"variance_gamma_european_put", MODEL_VARIANCE_GAMMA,
     CONTRACT_EUROPEAN_PUT, "equity_with_jumps", "VarianceGamma1dim",
     "PutEuro", "AP_Carr_VG", 8},
    {"normal_inverse_gaussian_european_call", MODEL_NORMAL_INVERSE_GAUSSIAN,
     CONTRACT_EUROPEAN_CALL, "equity_with_jumps", "NIG1dim", "CallEuro",
     "AP_Carr_NIG", 8},
    {"normal_inverse_gaussian_european_put", MODEL_NORMAL_INVERSE_GAUSSIAN,
     CONTRACT_EUROPEAN_PUT, "equity_with_jumps", "NIG1dim", "PutEuro",
     "AP_Carr_NIG", 8},
    {"cev_european_call", MODEL_CEV, CONTRACT_EUROPEAN_CALL,
     "equity_Black_Scholes_type", "cev1d", "CallEuro", "AP_BGM_Cev", 7},
    {"cev_european_put", MODEL_CEV, CONTRACT_EUROPEAN_PUT,
     "equity_Black_Scholes_type", "cev1d", "PutEuro", "AP_BGM_Cev", 7},
    {"schobel_zhu_european_call", MODEL_STEIN, CONTRACT_EUROPEAN_CALL,
     "equity_stochastic_volatility", "Stein1dim", "CallEuro",
     "AP_AntonelliScarlatti_Stein", 10},
    {"schobel_zhu_european_put", MODEL_STEIN, CONTRACT_EUROPEAN_PUT,
     "equity_stochastic_volatility", "Stein1dim", "PutEuro",
     "AP_AntonelliScarlatti_Stein", 10},
    {"vasicek_zero_coupon_call", MODEL_VASICEK, CONTRACT_ZERO_COUPON_CALL,
     "interest", "Vasicek1d", "ZeroCouponCallBondEuro",
     "CF_Vasicek1d_ZBCallEuro", 8},
    {"vasicek_zero_coupon_put", MODEL_VASICEK, CONTRACT_ZERO_COUPON_PUT,
     "interest", "Vasicek1d", "ZeroCouponPutBondEuro",
     "CF_Vasicek1d_ZBPutEuro", 8},
    {"vasicek_cap", MODEL_VASICEK, CONTRACT_CAP, "interest", "Vasicek1d",
     "Cap", "CF_Vasicek1d_Cap", 9},
    {"vasicek_floor", MODEL_VASICEK, CONTRACT_FLOOR, "interest", "Vasicek1d",
     "Floor", "CF_Vasicek1d_Floor", 9},
    {"ornstein_uhlenbeck_zero_coupon_call", MODEL_ORNSTEIN_UHLENBECK,
     CONTRACT_ZERO_COUPON_CALL, "interest", "Vasicek1d",
     "ZeroCouponCallBondEuro", "CF_Vasicek1d_ZBCallEuro", 8},
    {"ornstein_uhlenbeck_zero_coupon_put", MODEL_ORNSTEIN_UHLENBECK,
     CONTRACT_ZERO_COUPON_PUT, "interest", "Vasicek1d",
     "ZeroCouponPutBondEuro", "CF_Vasicek1d_ZBPutEuro", 8},
    {"ornstein_uhlenbeck_cap", MODEL_ORNSTEIN_UHLENBECK, CONTRACT_CAP,
     "interest", "Vasicek1d", "Cap", "CF_Vasicek1d_Cap", 9},
    {"ornstein_uhlenbeck_floor", MODEL_ORNSTEIN_UHLENBECK, CONTRACT_FLOOR,
     "interest", "Vasicek1d", "Floor", "CF_Vasicek1d_Floor", 9},
    {"hull_white_nelson_siegel_zero_coupon_call", MODEL_HULL_WHITE,
     CONTRACT_ZERO_COUPON_CALL, "interest", "HullWhite1d",
     "ZeroCouponCallBondEuro", "CF_HullWhite1d_ZBCallEuro", 10},
    {"hull_white_nelson_siegel_zero_coupon_put", MODEL_HULL_WHITE,
     CONTRACT_ZERO_COUPON_PUT, "interest", "HullWhite1d",
     "ZeroCouponPutBondEuro", "CF_HullWhite1d_ZBPutEuro", 10},
    {"hull_white_svensson_zero_coupon_call", MODEL_HULL_WHITE,
     CONTRACT_ZERO_COUPON_CALL, "interest", "HullWhite1d",
     "ZeroCouponCallBondEuro", "CF_HullWhite1d_ZBCallEuro", 12},
    {"hull_white_svensson_zero_coupon_put", MODEL_HULL_WHITE,
     CONTRACT_ZERO_COUPON_PUT, "interest", "HullWhite1d",
     "ZeroCouponPutBondEuro", "CF_HullWhite1d_ZBPutEuro", 12},
    {"g2_plus_plus_nelson_siegel_zero_coupon_call", MODEL_G2_PLUS_PLUS,
     CONTRACT_ZERO_COUPON_CALL, "interest", "HullWhite2d",
     "ZeroCouponCallBondEuro", "CF_ZBCallEuroHW2D", 13},
    {"g2_plus_plus_nelson_siegel_zero_coupon_put", MODEL_G2_PLUS_PLUS,
     CONTRACT_ZERO_COUPON_PUT, "interest", "HullWhite2d",
     "ZeroCouponPutBondEuro", "CF_ZBPutEuroHW2D", 13},
    {"g2_plus_plus_svensson_zero_coupon_call", MODEL_G2_PLUS_PLUS,
     CONTRACT_ZERO_COUPON_CALL, "interest", "HullWhite2d",
     "ZeroCouponCallBondEuro", "CF_ZBCallEuroHW2D", 15},
    {"g2_plus_plus_svensson_zero_coupon_put", MODEL_G2_PLUS_PLUS,
     CONTRACT_ZERO_COUPON_PUT, "interest", "HullWhite2d",
     "ZeroCouponPutBondEuro", "CF_ZBPutEuroHW2D", 15},
};

typedef struct PricingContext {
    const ModeSpec *spec;
    Model *model;
    Option *option;
    PricingMethod *method;
    VAR *model_variables;
    VAR *option_variables;
    NumFunc_1 *payoff;
    int error_result_index;
    char curve_filename[MAX_PATH];
} PricingContext;

static void set_scalar(VAR *variable, double value, void *owner) {
    switch (variable->Vtype) {
        case INT:
        case INT2:
        case RGINT13:
        case RGINT12:
        case RGINT130:
        case PINT:
        case BOOL:
        case PADE:
            variable->Val.V_INT = (int)llround(value);
            break;
        case LONG:
            variable->Val.V_LONG = (long)llround(value);
            break;
        default:
            variable->Val.V_DOUBLE = value;
            break;
    }
    if (variable->setter != NULL) variable->setter(owner);
}

static double annual_rate_percent(double continuous_rate) {
    return 100.0 * expm1(continuous_rate);
}

static const ModeSpec *find_mode(const char *mode) {
    const size_t count = sizeof(MODE_SPECS) / sizeof(MODE_SPECS[0]);
    for (size_t index = 0; index < count; ++index)
        if (strcmp(MODE_SPECS[index].mode, mode) == 0)
            return &MODE_SPECS[index];
    return NULL;
}

static void configure_method(PricingContext *context) {
    const int path_monte_carlo =
        strstr(context->method->Name, "Lookback") != NULL
        || strstr(context->method->Name, "FixedAsian_IS") != NULL
        || strstr(context->method->Name, "Kou_Out") != NULL
        || strstr(context->method->Name, "Kou_In") != NULL
        || strstr(context->method->Name, "WHBar_Kou") != NULL
        || strstr(context->method->Name, "Asian_Bates") != NULL
        || strstr(context->method->Name, "Bates_Out") != NULL
        || strstr(context->method->Name, "LongstaffSchwartz_Bates") != NULL;
    for (int index = 0; index < MAX_PAR; ++index) {
        VAR *parameter = &context->method->Par[index];
        if (parameter->Vtype == PREMIA_NULLTYPE) break;
        const char *name = parameter->Vname == NULL ? "" : parameter->Vname;
        if (strstr(name, "iterations") != NULL
            || strstr(name, "Iterations") != NULL
            || strstr(name, "Simulations") != NULL
            || strstr(name, "Samples") != NULL) {
            set_scalar(parameter,
                       (double)(path_monte_carlo
                           ? PREMIA_PATH_MC_ITERATIONS
                           : PREMIA_MC_ITERATIONS),
                       context->method);
        } else if (strstr(name, "discretization steps") != NULL
                   || strstr(name, "time steps") != NULL
                   || strstr(name, "TimeStepNumber") != NULL
                   || strstr(name, "SpaceStepNumber") != NULL) {
            set_scalar(parameter, 256.0, context->method);
        } else if (strstr(name, "Confidence Value") != NULL) {
            set_scalar(parameter, 0.95, context->method);
        } else if (strstr(name, "Exercise Dates") != NULL) {
            set_scalar(parameter, 64.0, context->method);
        } else if (strstr(name, "Monitoring Dates") != NULL) {
            set_scalar(parameter, 52.0, context->method);
        } else if (strstr(name, "Integration Points") != NULL) {
            set_scalar(parameter, 1024.0, context->method);
        }
    }
    context->error_result_index = -1;
    if (strcmp(context->method->Name, "MC_Merton") == 0)
        context->error_result_index = 2;
    else if (strcmp(context->method->Name, "MC_Kou_Digital_LRM") == 0)
        context->error_result_index = 1;
    else if (strcmp(context->method->Name, "MC_Merton_FixedLookback") == 0
             || strcmp(context->method->Name, "MC_Kou_LookbackFixed") == 0
             || strcmp(context->method->Name, "MC_Alfonsi_Asian_Bates") == 0
             || strcmp(context->method->Name, "MC_Alfonsi_Bates_Out") == 0)
        context->error_result_index = 2;
    else if (strcmp(
                 context->method->Name,
                 "MC_AM_Alfonsi_LongstaffSchwartz_Bates"
             ) == 0)
        context->error_result_index = 1;
    else if (strcmp(context->method->Name, "MC_FixedAsian_IS_Lelong") == 0
             || strcmp(context->method->Name, "MC_WHBar_Kou") == 0)
        context->error_result_index = 1;
    else if (strcmp(context->method->Name, "MC_Kou") == 0
             || strcmp(context->method->Name, "MC_Kou_Out_LRM") == 0
             || strcmp(context->method->Name, "MC_Kou_In_LRM") == 0)
        context->error_result_index = 2;
    else if (strcmp(context->method->Name, "MC_FixedAsian_ExactMethod") == 0
             || strcmp(context->method->Name, "MC_FixedAsian_KemnaVorst") == 0)
        context->error_result_index = 2;
}

static int initialize_context(
    const char *mode,
    const char *method_override,
    PricingContext *context
) {
    context->spec = find_mode(mode);
    if (context->spec == NULL) return 1;
    PremiaAsset *asset = premia_get_asset_from_name(context->spec->asset_name);
    if (asset == NULL) return 2;
    context->model = premia_get_model_from_name(
        asset, context->spec->model_name);
    if (context->model == NULL) return 3;
    context->option = premia_get_option_from_name(
        asset, context->model, context->spec->option_name);
    if (context->option == NULL) return 4;
    const char *method_name = method_override == NULL
        ? context->spec->method_name
        : method_override;
    context->method = premia_get_method_from_name(
        asset, context->model, context->option, method_name);
    if (context->method == NULL) return 5;
    context->model_variables = (VAR *)context->model->TypeModel;
    context->option_variables = (VAR *)context->option->TypeOpt;
    context->payoff = NULL;
    if (context->option_variables[0].Vtype == NUMFUNC_1)
        context->payoff = context->option_variables[0].Val.V_NUMFUNC_1;
    configure_method(context);
    return 0;
}

static int prepare_equity_model(
    PricingContext *context, const double *x
) {
    VAR *model = context->model_variables;
    switch (context->spec->model_kind) {
        case MODEL_BLACK_SCHOLES:
            if (x[0] <= 0.0 || x[3] <= 0.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], 0.0, context->model->TypeModel);
            set_scalar(&model[3], x[3], context->model->TypeModel);
            set_scalar(&model[4], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[5], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            return 0;
        case MODEL_HESTON:
            if (x[0] <= 0.0 || x[3] < 0.0 || x[4] <= 0.0
                || x[5] <= 0.0 || x[6] <= 0.0
                || x[7] <= -1.0 || x[7] >= 1.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[3], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            for (int index = 3; index <= 7; ++index)
                set_scalar(&model[index + 1], x[index],
                           context->model->TypeModel);
            return 0;
        case MODEL_BATES:
            if (x[0] <= 0.0 || x[3] < 0.0 || x[4] <= 0.0
                || x[5] <= 0.0 || x[6] <= 0.0
                || x[7] <= -1.0 || x[7] >= 1.0
                || x[8] < 0.0 || x[10] < 0.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[3], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            set_scalar(&model[4], x[3], context->model->TypeModel);
            set_scalar(&model[5], x[4], context->model->TypeModel);
            set_scalar(&model[6], x[5], context->model->TypeModel);
            set_scalar(&model[7], x[6], context->model->TypeModel);
            set_scalar(&model[8], x[8], context->model->TypeModel);
            set_scalar(&model[9], x[9], context->model->TypeModel);
            set_scalar(&model[10], x[10] * x[10], context->model->TypeModel);
            set_scalar(&model[11], x[7], context->model->TypeModel);
            return 0;
        case MODEL_MERTON:
            if (x[0] <= 0.0 || x[3] <= 0.0 || x[4] < 0.0
                || x[6] < 0.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], 0.0, context->model->TypeModel);
            set_scalar(&model[3], x[3], context->model->TypeModel);
            set_scalar(&model[4], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[5], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            set_scalar(&model[6], x[4], context->model->TypeModel);
            set_scalar(&model[7], x[5], context->model->TypeModel);
            set_scalar(&model[8], x[6] * x[6], context->model->TypeModel);
            return 0;
        case MODEL_KOU:
            if (x[0] <= 0.0 || x[3] <= 0.0 || x[4] < 0.0
                || x[5] <= 0.0 || x[5] >= 1.0
                || x[6] <= 1.0 || x[7] <= 0.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], 0.0, context->model->TypeModel);
            set_scalar(&model[3], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[4], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            set_scalar(&model[5], x[3], context->model->TypeModel);
            set_scalar(&model[6], x[4], context->model->TypeModel);
            set_scalar(&model[7], x[6], context->model->TypeModel);
            set_scalar(&model[8], x[7], context->model->TypeModel);
            set_scalar(&model[9], x[5], context->model->TypeModel);
            return 0;
        case MODEL_VARIANCE_GAMMA:
            if (x[0] <= 0.0 || x[3] <= 0.0 || x[4] <= 0.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], 0.0, context->model->TypeModel);
            set_scalar(&model[3], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[4], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            set_scalar(&model[5], x[3], context->model->TypeModel);
            set_scalar(&model[6], x[5], context->model->TypeModel);
            set_scalar(&model[7], x[4], context->model->TypeModel);
            return 0;
        case MODEL_NORMAL_INVERSE_GAUSSIAN: {
            if (x[0] <= 0.0 || x[3] <= fabs(x[4]) || x[5] <= 0.0)
                return 12;
            const double gamma = sqrt(x[3] * x[3] - x[4] * x[4]);
            const double sigma = sqrt(x[5] / gamma);
            const double theta = x[5] * x[4] / gamma;
            const double kappa = 1.0 / (x[5] * gamma);
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], 0.0, context->model->TypeModel);
            set_scalar(&model[3], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[4], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            set_scalar(&model[5], sigma, context->model->TypeModel);
            set_scalar(&model[6], theta, context->model->TypeModel);
            set_scalar(&model[7], kappa, context->model->TypeModel);
            return 0;
        }
        case MODEL_CEV:
            if (x[0] <= 0.0 || x[3] <= 0.0 || x[4] <= 0.0
                || x[4] >= 1.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            set_scalar(&model[3], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[4], x[3], context->model->TypeModel);
            set_scalar(&model[5], x[4], context->model->TypeModel);
            return 0;
        case MODEL_STEIN:
            if (x[0] <= 0.0 || x[4] <= 0.0 || x[6] <= 0.0
                || x[7] <= -1.0 || x[7] >= 1.0) return 12;
            set_scalar(&model[0], 0.0, context->model->TypeModel);
            set_scalar(&model[1], x[0], context->model->TypeModel);
            set_scalar(&model[2], annual_rate_percent(x[2]),
                       context->model->TypeModel);
            set_scalar(&model[3], annual_rate_percent(x[1]),
                       context->model->TypeModel);
            for (int index = 3; index <= 7; ++index)
                set_scalar(&model[index + 1], x[index],
                           context->model->TypeModel);
            return 0;
        default:
            return 15;
    }
}

static int prepare_rate_model(PricingContext *context, const double *x) {
    VAR *model = context->model_variables;
    if (x[1] <= 0.0 || x[2] <= 0.0) return 12;
    set_scalar(&model[0], 0.0, context->model->TypeModel);
    set_scalar(&model[1], x[0], context->model->TypeModel);
    set_scalar(&model[2], x[1], context->model->TypeModel);
    set_scalar(&model[3], x[2], context->model->TypeModel);
    set_scalar(&model[4], context->spec->model_kind == MODEL_VASICEK
        ? x[3] : 0.0, context->model->TypeModel);
    return 0;
}

static int uses_fitted_curve(const PricingContext *context) {
    return context->spec->model_kind == MODEL_HULL_WHITE
        || context->spec->model_kind == MODEL_G2_PLUS_PLUS;
}

static int is_svensson_curve(const PricingContext *context) {
    return strstr(context->spec->mode, "svensson") != NULL;
}

static double curve_discount(
    const PricingContext *context, const double *curve, double maturity
) {
    if (maturity == 0.0) return 1.0;
    const double x1 = maturity / curve[is_svensson_curve(context) ? 4 : 3];
    const double loading1 = -expm1(-x1) / x1;
    double zero_rate = curve[0] + curve[1] * loading1
        + curve[2] * (loading1 - exp(-x1));
    if (is_svensson_curve(context)) {
        const double x2 = maturity / curve[5];
        const double loading2 = -expm1(-x2) / x2;
        zero_rate += curve[3] * (loading2 - exp(-x2));
    }
    return exp(-maturity * zero_rate);
}

static void compact_number(char *output, size_t capacity, double value, int digits) {
    /* Premia's legacy curve reader accepts at most 19 characters per line. */
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.*g", digits, value);
    const char *start = buffer;
    if (buffer[0] == '0' && buffer[1] == '.') ++start;
    snprintf(output, capacity, "%s", start);
}

static int write_curve_node(
    FILE *file,
    const PricingContext *context,
    const double *curve,
    double maturity
) {
    char discount_text[16];
    char maturity_text[16];
    compact_number(
        discount_text,
        sizeof(discount_text),
        curve_discount(context, curve, maturity),
        7
    );
    compact_number(maturity_text, sizeof(maturity_text), maturity, 7);
    return fprintf(file, "%s t=%s\n", discount_text, maturity_text) < 0
        ? 16 : 0;
}

static int write_fitted_curve(
    PricingContext *context, const double *curve, double expiry, double maturity
) {
    FILE *file = fopen(context->curve_filename, "w");
    if (file == NULL) return 16;
    /*
     * Premia linearly interpolates external curves.  Tight brackets around
     * each contractual date keep that interpolation local while the stored
     * discounts remain independent Nelson-Siegel/Svensson evaluations.
     */
    const double resolution = 1.0e-4;
    const double expiry_lower = floor(expiry / resolution) * resolution;
    const double expiry_upper = expiry_lower + resolution;
    const double maturity_lower = floor(maturity / resolution) * resolution;
    const double maturity_upper = maturity_lower + resolution;
    int status = fprintf(file, "1 t=0\n") < 0 ? 16 : 0;
    if (status == 0 && expiry_lower > 0.0)
        status = write_curve_node(file, context, curve, expiry_lower);
    if (status == 0)
        status = write_curve_node(file, context, curve, expiry_upper);
    if (status == 0 && maturity_lower > expiry_upper)
        status = write_curve_node(file, context, curve, maturity_lower);
    if (status == 0)
        status = write_curve_node(file, context, curve, maturity_upper);
    const int close_status = fclose(file);
    return status != 0 || close_status != 0 ? 16 : 0;
}

static int prepare_fitted_rate_model(
    PricingContext *context, const double *x
) {
    VAR *model = context->model_variables;
    const int model_count = context->spec->model_kind == MODEL_HULL_WHITE ? 2 : 5;
    const int curve_count = is_svensson_curve(context) ? 6 : 4;
    const double *curve = &x[model_count];
    const int product_offset = model_count + curve_count;
    const double expiry = x[product_offset + 1];
    const double maturity = x[product_offset + 2];
    const double first_tau = curve[is_svensson_curve(context) ? 4 : 3];
    const double second_tau = is_svensson_curve(context) ? curve[5] : 1.0;
    if (first_tau <= 0.0 || second_tau <= 0.0
        || expiry <= 0.0 || maturity <= expiry) return 12;

    model[1].Val.V_ENUM.value = 1;
    VAR *curve_parameter = lookup_premia_enum_par(&model[1], 1);
    if (curve_parameter == NULL || curve_parameter[0].Val.V_FILENAME == NULL)
        return 13;
    strcpy(curve_parameter[0].Val.V_FILENAME, context->curve_filename);

    set_scalar(&model[0], 0.0, context->model->TypeModel);
    if (context->spec->model_kind == MODEL_HULL_WHITE) {
        if (x[0] <= 0.0 || x[1] <= 0.0) return 12;
        set_scalar(&model[2], x[0], context->model->TypeModel);
        set_scalar(&model[3], x[1], context->model->TypeModel);
    } else {
        const double a = x[0];
        const double sigma = x[1];
        const double b = x[2];
        const double eta = x[3];
        const double rho = x[4];
        const double delta = a - b;
        if (a <= 0.0 || sigma <= 0.0 || b <= 0.0 || eta <= 0.0
            || rho <= -1.0 || rho >= 1.0 || fabs(delta) <= 1.0e-8)
            return 12;
        /*
         * If z=x+y is the G2++ stochastic short rate, then
         * dz=-a*z*dt+u*dt+sigma_r*dW_r with u=(a-b)*y.  The following
         * volatility and correlation map that exact system to Premia HW2D.
         */
        const double sigma_r = sqrt(
            sigma * sigma + eta * eta + 2.0 * rho * sigma * eta
        );
        const double sigma_u = fabs(delta) * eta;
        double rho_ru = (rho * sigma + eta) / sigma_r;
        if (delta < 0.0) rho_ru = -rho_ru;
        if (rho_ru < -1.0) rho_ru = -1.0;
        if (rho_ru > 1.0) rho_ru = 1.0;
        set_scalar(&model[2], 0.0, context->model->TypeModel);
        set_scalar(&model[3], a, context->model->TypeModel);
        set_scalar(&model[4], sigma_r, context->model->TypeModel);
        set_scalar(&model[5], b, context->model->TypeModel);
        set_scalar(&model[6], sigma_u, context->model->TypeModel);
        set_scalar(&model[7], rho_ru, context->model->TypeModel);
    }
    return write_fitted_curve(context, curve, expiry, maturity);
}

static int prepare_contract(PricingContext *context, const double *x) {
    const int count = context->spec->field_count;
    switch (context->spec->contract_kind) {
        case CONTRACT_EUROPEAN_CALL:
        case CONTRACT_EUROPEAN_PUT:
        case CONTRACT_AMERICAN_CALL:
        case CONTRACT_AMERICAN_PUT:
            if (context->payoff == NULL) return 13;
            set_scalar(&context->payoff->Par[0], x[count - 2],
                       context->payoff);
            set_scalar(&context->option_variables[1], x[count - 1],
                       context->option->TypeOpt);
            break;
        case CONTRACT_DIGITAL_CALL:
            if (context->payoff == NULL) return 13;
            set_scalar(&context->payoff->Par[0], x[count - 2],
                       context->payoff);
            set_scalar(&context->payoff->Par[1], 1.0, context->payoff);
            set_scalar(&context->option_variables[1], x[count - 1],
                       context->option->TypeOpt);
            break;
        case CONTRACT_SINGLE_BARRIER:
            {
            const int offset = count - 3;
            NumFunc_1 *limit = context->option_variables[1].Val.V_NUMFUNC_1;
            NumFunc_1 *payoff = context->option_variables[2].Val.V_NUMFUNC_1;
            NumFunc_1 *rebate = context->option_variables[3].Val.V_NUMFUNC_1;
            if (limit == NULL || payoff == NULL || rebate == NULL) return 13;
            set_scalar(&context->option_variables[0], x[offset + 1],
                       context->option->TypeOpt);
            set_scalar(&limit->Par[0], 0.0, limit);
            set_scalar(&limit->Par[1], x[offset + 1], limit);
            set_scalar(&limit->Par[2], 1.0, limit);
            set_scalar(&limit->Par[3], x[offset + 2], limit);
            set_scalar(&payoff->Par[0], x[offset], payoff);
            set_scalar(&rebate->Par[0], 0.0, rebate);
            break;
            }
        case CONTRACT_SINGLE_BARRIER_REBATE:
            {
            const int offset = count - 4;
            NumFunc_1 *limit = context->option_variables[1].Val.V_NUMFUNC_1;
            NumFunc_1 *payoff = context->option_variables[2].Val.V_NUMFUNC_1;
            NumFunc_1 *rebate = context->option_variables[3].Val.V_NUMFUNC_1;
            if (limit == NULL || payoff == NULL || rebate == NULL) return 13;
            set_scalar(&context->option_variables[0], x[offset + 1],
                       context->option->TypeOpt);
            set_scalar(&limit->Par[0], 0.0, limit);
            set_scalar(&limit->Par[1], x[offset + 1], limit);
            set_scalar(&limit->Par[2], 1.0, limit);
            set_scalar(&limit->Par[3], x[offset + 2], limit);
            set_scalar(&payoff->Par[0], x[offset], payoff);
            set_scalar(&rebate->Par[0], x[offset + 3], rebate);
            break;
            }
        case CONTRACT_DOUBLE_BARRIER:
            {
            const int offset = count - 4;
            NumFunc_1 *payoff = context->option_variables[0].Val.V_NUMFUNC_1;
            NumFunc_1 *rebate = context->option_variables[1].Val.V_NUMFUNC_1;
            NumFunc_1 *lower = context->option_variables[2].Val.V_NUMFUNC_1;
            NumFunc_1 *upper = context->option_variables[3].Val.V_NUMFUNC_1;
            if (payoff == NULL || rebate == NULL
                || lower == NULL || upper == NULL) return 13;
            set_scalar(&payoff->Par[0], x[offset], payoff);
            set_scalar(&rebate->Par[0], 0.0, rebate);
            set_scalar(&lower->Par[0], x[offset + 2], lower);
            set_scalar(&upper->Par[0], x[offset + 3], upper);
            set_scalar(&context->option_variables[4], x[offset + 1],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[5], 0.0,
                       context->option->TypeOpt);
            break;
            }
        case CONTRACT_FIXED_LOOKBACK_CALL:
            {
            const int offset = count - 2;
            NumFunc_2 *payoff = context->option_variables[1].Val.V_NUMFUNC_2;
            NumFunc_2 *path = context->option_variables[2].Val.V_NUMFUNC_2;
            if (payoff == NULL || path == NULL) return 13;
            set_scalar(&context->option_variables[0], x[offset + 1],
                       context->option->TypeOpt);
            set_scalar(&payoff->Par[0], x[offset], payoff);
            set_scalar(&path->Par[0], 0.0, path);
            set_scalar(&path->Par[1], x[offset + 1], path);
            set_scalar(&path->Par[2], 1.0, path);
            set_scalar(&path->Par[3], x[0], path);
            set_scalar(&path->Par[4], x[0], path);
            break;
            }
        case CONTRACT_FIXED_ASIAN:
            {
            const int has_explicit_monitoring_count =
                context->spec->model_kind == MODEL_KOU;
            const int offset = count - (has_explicit_monitoring_count ? 3 : 2);
            NumFunc_2 *payoff = context->option_variables[1].Val.V_NUMFUNC_2;
            NumFunc_2 *path = context->option_variables[2].Val.V_NUMFUNC_2;
            if (payoff == NULL || path == NULL) return 13;
            set_scalar(&context->option_variables[0], x[offset + 1],
                       context->option->TypeOpt);
            set_scalar(&payoff->Par[0], x[offset], payoff);
            set_scalar(&path->Par[0], 0.0, path);
            set_scalar(&path->Par[1], x[offset + 1], path);
            set_scalar(&path->Par[2], 1.0, path);
            set_scalar(&path->Par[3], x[0], path);
            set_scalar(&path->Par[4], x[0], path);
            set_scalar(&context->option_variables[6], 1.0,
                       context->option->TypeOpt);
            if (has_explicit_monitoring_count) {
                for (int index = 0; index < MAX_PAR; ++index) {
                    VAR *parameter = &context->method->Par[index];
                    if (parameter->Vtype == PREMIA_NULLTYPE) break;
                    const char *name = parameter->Vname == NULL
                        ? "" : parameter->Vname;
                    if (strstr(name, "Monitoring Dates") != NULL) {
                        set_scalar(parameter, x[count - 1], context->method);
                        break;
                    }
                }
            }
            break;
            }
        case CONTRACT_ZERO_COUPON_CALL:
        case CONTRACT_ZERO_COUPON_PUT:
            {
            const int offset = count - 4;
            if (context->payoff == NULL) return 13;
            set_scalar(&context->payoff->Par[0], x[offset], context->payoff);
            set_scalar(&context->option_variables[2], x[offset + 1],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[3], x[offset + 2],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[4], x[offset + 3],
                       context->option->TypeOpt);
            break;
            }
        case CONTRACT_CAP:
        case CONTRACT_FLOOR:
            {
            const int offset = count - 5;
            set_scalar(&context->option_variables[3], x[offset + 3],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[4], x[offset],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[5], 100.0 * x[offset + 1],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[6], x[offset + 4],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[7], x[offset + 2],
                       context->option->TypeOpt);
            set_scalar(&context->option_variables[8], 1.0,
                       context->option->TypeOpt);
            break;
            }
    }
    return 0;
}

static int price_row(
    PricingContext *context,
    const double *x,
    int count,
    double *price,
    double *standard_error
) {
    if (count != context->spec->field_count) return 10;
    for (int index = 0; index < count; ++index)
        if (!isfinite(x[index])) return 11;
    int status;
    if (uses_fitted_curve(context))
        status = prepare_fitted_rate_model(context, x);
    else if (context->spec->model_kind == MODEL_VASICEK
             || context->spec->model_kind == MODEL_ORNSTEIN_UHLENBECK)
        status = prepare_rate_model(context, x);
    else
        status = prepare_equity_model(context, x);
    if (status != 0) {
        if (uses_fitted_curve(context)) remove(context->curve_filename);
        return status;
    }
    status = prepare_contract(context, x);
    if (status != 0) {
        if (uses_fitted_curve(context)) remove(context->curve_filename);
        return status;
    }
    status = context->method->Compute(
        context->option->TypeOpt,
        context->model->TypeModel,
        context->method
    );
    if (uses_fitted_curve(context)) remove(context->curve_filename);
    if (status != 0) return status;
    *price = context->method->Res[0].Val.V_DOUBLE;
    *standard_error = context->error_result_index < 0
        ? 0.0
        : fabs(context->method->Res[context->error_result_index].Val.V_DOUBLE);
    return isfinite(*price) && isfinite(*standard_error) ? 0 : 14;
}

static int parse_line(
    char *line, char **row_id, double *values, int *value_count
) {
    char *token = strtok(line, "\t \r\n");
    if (token == NULL) return 1;
    *row_id = token;
    int count = 0;
    while ((token = strtok(NULL, "\t \r\n")) != NULL) {
        if (count == MAX_NUMERIC_FIELDS) return 2;
        char *end = NULL;
        errno = 0;
        values[count] = strtod(token, &end);
        if (errno != 0 || end == token || *end != '\0') return 3;
        ++count;
    }
    *value_count = count;
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: premia_runner.exe PREMIA_ROOT MODE [METHOD]\n");
        return 2;
    }
    premia_self_set_global_vars(argv[1]);
    if (InitErrorMsg() != 0 || InitVar() != 0) return 3;

    PricingContext context = {0};
    const int initialization_status = initialize_context(
        argv[2], argc == 4 ? argv[3] : NULL, &context);
    if (initialization_status != 0) {
        fprintf(stderr, "Premia context initialization failed with status %d.\n",
                initialization_status);
        ExitVar();
        return 4;
    }
    snprintf(
        context.curve_filename,
        sizeof(context.curve_filename),
        "ai_factory_validation_curve_%lu.dat",
        (unsigned long)GetCurrentProcessId()
    );

    char line[INPUT_LINE_CAPACITY];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        char *row_id = NULL;
        double values[MAX_NUMERIC_FIELDS];
        int value_count = 0;
        if (parse_line(line, &row_id, values, &value_count) != 0) {
            fprintf(stderr, "Malformed Premia input row.\n");
            ExitVar();
            return 5;
        }
        double price = NAN;
        double standard_error = NAN;
        const int status = price_row(
            &context, values, value_count, &price, &standard_error);
        printf("RESULT\t%s\t%d\t%.17g\t%.17g\n",
               row_id, status, price, standard_error);
    }
    ExitVar();
    return 0;
}
