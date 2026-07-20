#pragma once

#include <cstddef>
#include <cstdint>

namespace ai_factory::cuda {

struct HestonRow {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double initial_variance;
    double kappa;
    double theta;
    double volatility_of_variance;
    double rho;
    double strike;
    double maturity;
    std::uint64_t seed;
    int scheme;
};

struct RoughBergomiRow {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double forward_variance;
    double eta;
    double alpha;
    double rho;
    double strike;
    double maturity;
    std::uint64_t seed;
};

constexpr std::size_t kRoughHestonFactorCount = 8U;

struct RoughHestonRow {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double initial_variance;
    double kappa;
    double theta;
    double volatility_of_variance;
    double hurst;
    double rho;
    double strike;
    double maturity;
    std::uint64_t seed;
};

struct BlackScholesRow {
    double spot;
    double risk_free_rate;
    double dividend_yield;
    double volatility;
    double strike;
    double maturity;
    std::uint64_t seed;
};

struct MonteCarloOutput {
    double price;
    double standard_error;
};

struct PriceDeltaOutput {
    double price;
    double standard_error;
    double delta;
    double delta_standard_error;
};

struct AutocallTerms {
    double autocall_barrier;
    double coupon_barrier;
    double protection_barrier;
    double coupon_rate;
    std::size_t observation_count;
    std::size_t first_autocall_observation;
};

struct BlackScholesAutocallRow {
    BlackScholesRow model;
    AutocallTerms product;
};

struct HestonAutocallRow {
    HestonRow model;
    AutocallTerms product;
};

struct RoughBergomiAutocallRow {
    RoughBergomiRow model;
    AutocallTerms product;
};

struct RoughHestonAutocallRow {
    RoughHestonRow model;
    AutocallTerms product;
};

struct BarrierTerms {
    double barrier;
};

struct BlackScholesBarrierRow {
    BlackScholesRow model;
    BarrierTerms product;
};

struct HestonBarrierRow {
    HestonRow model;
    BarrierTerms product;
};

struct RoughBergomiBarrierRow {
    RoughBergomiRow model;
    BarrierTerms product;
};

struct RoughHestonBarrierRow {
    RoughHestonRow model;
    BarrierTerms product;
};

struct SwapTerms {
    double start_time;
    double accrual_period;
    double fixed_rate;
    double notional;
    int payment_count;
    int direction;
};

struct ZeroCouponBondTerms {
    double maturity;
    double notional;
};

struct SwaptionTerms {
    double expiry;
    double accrual_period;
    double fixed_rate;
    double notional;
    int payment_count;
    int direction;
};

struct CapletTerms {
    double fixing_time;
    double accrual_period;
    double strike;
    double notional;
};

struct Black76Model {
    double volatility;
    double displacement;
};

struct Black76Curve {
    double beta0;
    double beta1;
    double beta2;
    double tau;
};

struct Black76CapletRow {
    Black76Model model;
    Black76Curve curve;
    CapletTerms product;
};

struct Black76SwaptionRow {
    Black76Model model;
    Black76Curve curve;
    SwaptionTerms product;
};

struct BermudanSwaptionTerms {
    double first_exercise;
    double exercise_period;
    double accrual_period;
    double fixed_rate;
    double notional;
    int exercise_count;
    int payment_count;
    int direction;
};

struct HullWhiteSwapRow {
    double mean_reversion;
    double volatility;
    double beta0;
    double beta1;
    double beta2;
    double tau;
    SwapTerms product;
};

struct HullWhiteZeroCouponBondRow {
    double mean_reversion;
    double volatility;
    double beta0;
    double beta1;
    double beta2;
    double tau;
    ZeroCouponBondTerms product;
};

struct HullWhiteSwaptionRow {
    double mean_reversion;
    double volatility;
    double beta0;
    double beta1;
    double beta2;
    double tau;
    SwaptionTerms product;
    std::uint64_t seed;
};

struct HullWhiteCapletRow {
    double mean_reversion;
    double volatility;
    double beta0;
    double beta1;
    double beta2;
    double tau;
    CapletTerms product;
    std::uint64_t seed;
};

struct HullWhiteBermudanSwaptionRow {
    double mean_reversion;
    double volatility;
    double beta0;
    double beta1;
    double beta2;
    double tau;
    BermudanSwaptionTerms product;
    std::uint64_t seed;
};

struct CirModel {
    double initial_rate;
    double kappa;
    double theta;
    double volatility;
};

struct CirSwapRow {
    CirModel model;
    SwapTerms product;
};

struct CirZeroCouponBondRow {
    CirModel model;
    ZeroCouponBondTerms product;
};

struct CirSwaptionRow {
    CirModel model;
    SwaptionTerms product;
    std::uint64_t seed;
};

struct CirCapletRow {
    CirModel model;
    CapletTerms product;
    std::uint64_t seed;
};

struct CirBermudanSwaptionRow {
    CirModel model;
    BermudanSwaptionTerms product;
    std::uint64_t seed;
};

struct CirPlusPlusModel {
    double initial_factor;
    double kappa;
    double theta;
    double volatility;
};

struct CirPlusPlusCurve {
    double beta0;
    double beta1;
    double beta2;
    double tau;
};

struct CirPlusPlusSwapRow {
    CirPlusPlusModel model;
    CirPlusPlusCurve curve;
    SwapTerms product;
};

struct CirPlusPlusZeroCouponBondRow {
    CirPlusPlusModel model;
    CirPlusPlusCurve curve;
    ZeroCouponBondTerms product;
};

struct CirPlusPlusSwaptionRow {
    CirPlusPlusModel model;
    CirPlusPlusCurve curve;
    SwaptionTerms product;
    std::uint64_t seed;
};

struct CirPlusPlusCapletRow {
    CirPlusPlusModel model;
    CirPlusPlusCurve curve;
    CapletTerms product;
    std::uint64_t seed;
};

struct CirPlusPlusBermudanSwaptionRow {
    CirPlusPlusModel model;
    CirPlusPlusCurve curve;
    BermudanSwaptionTerms product;
    std::uint64_t seed;
};

struct G2PlusPlusModel {
    double mean_reversion_x;
    double volatility_x;
    double mean_reversion_y;
    double volatility_y;
    double rho;
};

struct G2PlusPlusCurve {
    double beta0;
    double beta1;
    double beta2;
    double tau;
};

struct G2PlusPlusSwapRow {
    G2PlusPlusModel model;
    G2PlusPlusCurve curve;
    SwapTerms product;
};

struct G2PlusPlusZeroCouponBondRow {
    G2PlusPlusModel model;
    G2PlusPlusCurve curve;
    ZeroCouponBondTerms product;
};

struct G2PlusPlusSwaptionRow {
    G2PlusPlusModel model;
    G2PlusPlusCurve curve;
    SwaptionTerms product;
    std::uint64_t seed;
};

struct G2PlusPlusCapletRow {
    G2PlusPlusModel model;
    G2PlusPlusCurve curve;
    CapletTerms product;
    std::uint64_t seed;
};

struct G2PlusPlusBermudanSwaptionRow {
    G2PlusPlusModel model;
    G2PlusPlusCurve curve;
    BermudanSwaptionTerms product;
    std::uint64_t seed;
};

struct AutocallOutput {
    double price;
    double standard_error;
    double autocall_probability;
    double mean_autocall_time;
    double maturity_probability;
    double coupon_payment_frequency;
    double mean_total_coupon;
    double capital_loss_probability;
    double mean_redemption_given_loss;
};

struct CudaTiming {
    float simulation_ms;
    float total_ms;
};

struct HestonCudaWorkspace;
struct RoughBergomiCudaWorkspace;
struct BlackScholesCudaWorkspace;

}  // namespace ai_factory::cuda
