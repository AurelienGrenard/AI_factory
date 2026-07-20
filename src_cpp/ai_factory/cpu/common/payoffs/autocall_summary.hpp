#pragma once

#include "ai_factory/cuda/common/types.cuh"
#include "ai_factory/cuda/common/autocall.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ai_factory::cpu::payoffs {

struct AutocallSums {
    double price = 0.0;
    double price_squared = 0.0;
    double autocall = 0.0;
    double autocall_time = 0.0;
    double coupon_payment_frequency = 0.0;
    double total_coupon = 0.0;
    double capital_loss = 0.0;
    double loss_redemption = 0.0;
};

inline void add(
    AutocallSums& sums,
    const cuda::autocall_detail::PathMetrics& metrics
) {
    sums.price += metrics.discounted_payoff;
    sums.price_squared += metrics.discounted_payoff * metrics.discounted_payoff;
    sums.autocall += metrics.autocall;
    sums.autocall_time += metrics.autocall_time;
    sums.coupon_payment_frequency += metrics.coupon_payment_frequency;
    sums.total_coupon += metrics.total_coupon;
    sums.capital_loss += metrics.capital_loss;
    sums.loss_redemption += metrics.loss_redemption;
}

inline cuda::AutocallOutput summarize(
    const AutocallSums& sums,
    std::size_t num_paths
) {
    const double count = static_cast<double>(num_paths);
    const double price = sums.price / count;
    const double variance =
        (sums.price_squared - count * price * price)
        / static_cast<double>(num_paths - 1U);
    const double autocall_probability = sums.autocall / count;
    const double loss_probability = sums.capital_loss / count;
    return {
        price,
        std::sqrt(std::max(variance, 0.0)) / std::sqrt(count),
        autocall_probability,
        sums.autocall > 0.0 ? sums.autocall_time / sums.autocall : 0.0,
        1.0 - autocall_probability,
        sums.coupon_payment_frequency / count,
        sums.total_coupon / count,
        loss_probability,
        sums.capital_loss > 0.0
            ? sums.loss_redemption / sums.capital_loss
            : 0.0,
    };
}

}  // namespace ai_factory::cpu::payoffs
