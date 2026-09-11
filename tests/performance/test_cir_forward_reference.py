"""CPU checks for the independent CIR method-exploration reference."""
import math
import unittest
import numpy as np
from scipy.integrate import solve_ivp
import QuantLib as ql
from validation.quantlib.model.fixed_income.cir.reference import quantlib_model

from tools.performance.cir_forward_reference import bond, bond_coefficients, forward_transition, pde_price


class CirForwardReferenceTest(unittest.TestCase):
    models = [dict(mean_reversion=k, long_term_mean=theta, volatility=sigma, initial_state=r)
              for k, theta, sigma, r in ((.6,.04,.15,.03), (.05,.015,.12,.001),
                                       (1.5,.1,.05,.2), (.03,.001,.05,0.))]

    def test_forward_moments_against_independent_drift_ode(self):
        for model in self.models:
            for interval, remaining in ((1/504, 20), (.5, 2), (10, 0)):
                k, theta, sigma, r = (model[key] for key in
                    ('mean_reversion','long_term_mean','volatility','initial_state'))
                def rhs(t, state):
                    a = k + sigma*sigma*bond_coefficients(model, interval + remaining-t)[1]
                    return [k*theta-a*state[0], (2*k*theta+sigma*sigma)*state[0]-2*a*state[1]]
                expected = solve_ivp(rhs, (0, interval), (r,r*r), rtol=1e-11, atol=1e-14).y[:,-1]
                scale, degree, loading = forward_transition(model, interval, remaining)
                mean = scale*(degree+loading*r)
                variance = 2*scale*scale*(degree+2*loading*r)
                self.assertAlmostEqual(mean, expected[0], delta=2e-10)
                self.assertAlmostEqual(variance+mean*mean, expected[1], delta=2e-10)

    def test_bonds_against_quantlib(self):
        for model in self.models:
            reference = quantlib_model(model, None, {})
            for maturity in (.01, .5, 5, 40):
                self.assertAlmostEqual(float(bond(model, model['initial_state'], maturity)),
                                       reference.discountBond(0., maturity, model['initial_state']), delta=2e-12)

    def test_pde_single_payment_against_bond_option(self):
        product = dict(notional=1., strike=.04, accrual_fraction=.5, first_exercise_time=126,
                       payment_interval=126, payment_count=1, exercise_count=1)
        for model in self.models:
            reference = quantlib_model(model, None, {})
            coefficient = 1 + product['strike']*product['accrual_fraction']
            for side, kind in [('payer', ql.Option.Put), ('receiver', ql.Option.Call)]:
                exact = coefficient*reference.discountBondOption(kind, 1/coefficient, .5, 1.)
                numerical = pde_price(model, product, side, nodes=1024, steps_per_year=256)
                self.assertAlmostEqual(numerical, exact, delta=2e-6)


if __name__ == '__main__':
    unittest.main()
