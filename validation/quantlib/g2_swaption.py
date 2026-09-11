"""Independent QuantLib European G2 swaption diagnostic, never a CUDA pricer.

Uses QuantLib's native G2SwaptionEngine (conditional Gaussian integration).
It is an explicit reference calculation, not automatic cache certification or
a claim that the ordered Premia candidate inventory has been exhausted.
"""
from __future__ import annotations

import math
import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.swaption import swaption_times
from validation.quantlib.term_structure import DAY_COUNTER, REFERENCE_DATE, date_from_time


def swaption_price(model, product, side, *, integration_points=128, integration_range=9.0):
    """Price the regular physical swaption, preserving its contractual accrual."""
    if side not in {'payer', 'receiver'}:
        raise ValueError('Swaption side must be payer or receiver')
    notional = positive_number(product, 'notional', 'G2 swaption')
    accrual = positive_number(product, 'accrual_fraction', 'G2 swaption')
    interval = positive_number(product, 'payment_interval', 'G2 swaption')
    strike = finite_number(product, 'strike', 'G2 swaption')
    times = swaption_times(product)
    if strike < 0 or integration_points < 16 or integration_range <= 0:
        raise ValueError('Invalid G2 swaption reference configuration')
    # A constant strike rescaling matches all fixed coupons even when alpha
    # differs from the model clock interval; floating-leg telescoping is unchanged.
    fixed_rate = strike * accrual / interval
    schedule = ql.Schedule([date_from_time(t) for t in times], ql.NullCalendar(), ql.Unadjusted)
    curve = model.termStructure()
    index = ql.IborIndex('AI Factory synthetic', ql.Period(round(interval*252), ql.Days),
        0, ql.USDCurrency(), ql.NullCalendar(), ql.Unadjusted, False, DAY_COUNTER, curve)
    swap_type = ql.VanillaSwap.Payer if side == 'payer' else ql.VanillaSwap.Receiver
    swap = ql.VanillaSwap(swap_type, notional, schedule, fixed_rate, DAY_COUNTER,
        schedule, index, 0.0, DAY_COUNTER)
    swaption = ql.Swaption(swap, ql.EuropeanExercise(date_from_time(times[0])))
    previous = ql.Settings.instance().evaluationDate
    try:
        ql.Settings.instance().evaluationDate = REFERENCE_DATE
        swaption.setPricingEngine(ql.G2SwaptionEngine(model, integration_range, integration_points))
        return float(swaption.NPV())
    finally:
        ql.Settings.instance().evaluationDate = previous


def monte_carlo_prices(model, product, *, paths=1 << 18, seed=2026090801):
    """Independent CPU diagnostic under the exercise-date bond measure.

    Gaussian exponential tilting shifts the factor mean by -Cov(factor, I).
    SciPy quadrature computes those covariances; QuantLib supplies each affine
    bond coefficient. PCG64, NumPy FP64 and this measure differ from CUDA's Q
    joint-integral simulation. Returns payer/receiver (price, standard error).
    """
    import numpy as np
    from scipy.integrate import quad

    if paths < 2:
        raise ValueError('Monte Carlo requires at least two independent paths')
    a, sigma, b, eta, rho = list(model.params())
    times = swaption_times(product)
    expiry = times[0]
    loading = lambda k, t: -math.expm1(-k*t)/k
    covariance = np.array([[sigma*sigma*loading(2*a, expiry),
        rho*sigma*eta*loading(a+b, expiry)],
        [rho*sigma*eta*loading(a+b, expiry), eta*eta*loading(2*b, expiry)]])
    mean = -np.array([
        quad(lambda t: sigma*sigma*math.exp(-a*t)*loading(a,t)
            + rho*sigma*eta*math.exp(-a*t)*loading(b,t), 0, expiry, epsabs=1e-13)[0],
        quad(lambda t: eta*eta*math.exp(-b*t)*loading(b,t)
            + rho*sigma*eta*math.exp(-b*t)*loading(a,t), 0, expiry, epsabs=1e-13)[0]])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-13:
        raise ValueError('Invalid Gaussian covariance')
    root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    log_a = np.array([math.log(model.discountBond(expiry, t, [0.0, 0.0])) for t in times[1:]])
    bx = np.array([loading(a, t-expiry) for t in times[1:]])
    by = np.array([loading(b, t-expiry) for t in times[1:]])
    coupons = np.full(len(log_a), product['strike']*product['accrual_fraction'])
    coupons[-1] += 1
    factor = product['notional']*model.termStructure().discount(expiry)
    random = np.random.Generator(np.random.PCG64(seed))
    sums, squares = np.zeros(2), np.zeros(2)
    for start in range(0, paths, 4096):
        states = random.standard_normal((min(4096, paths-start),2)) @ root.T + mean
        fixed_leg = np.exp(log_a - states[:, :1]*bx - states[:, 1:]*by) @ coupons
        swap = factor*(1-fixed_leg)
        values = np.array([np.maximum(swap,0), np.maximum(-swap,0)])
        sums += values.sum(axis=1)
        squares += (values*values).sum(axis=1)
    prices = sums/paths
    errors = np.sqrt(np.maximum(squares/paths-prices*prices,0)/(paths-1))
    return {side:(float(prices[i]),float(errors[i])) for i,side in enumerate(('payer','receiver'))}
