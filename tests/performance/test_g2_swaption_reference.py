"""Opt-in qualification of the independent European G2 diagnostic engines."""
import unittest
import QuantLib as ql
from validation.quantlib.g2_swaption import monte_carlo_prices, swaption_price
from validation.quantlib.term_structure import DAY_COUNTER, REFERENCE_DATE


class G2SwaptionReferenceTest(unittest.TestCase):
    def setUp(self):
        curve = ql.YieldTermStructureHandle(ql.FlatForward(REFERENCE_DATE, .03, DAY_COUNTER))
        self.model = ql.G2(curve, .1, .01, .3, .015, -.5)
        self.product = dict(notional=1., strike=.04, exercise_time=2.,
                            payment_interval=.5, accrual_fraction=.5, payment_count=10)

    def test_single_coupon_identity_and_evaluation_date(self):
        product = dict(self.product, payment_count=1)
        coefficient = 1+product['strike']*product['accrual_fraction']
        previous = ql.Settings.instance().evaluationDate
        for side, kind in [('payer', ql.Option.Put), ('receiver', ql.Option.Call)]:
            expected = coefficient*self.model.discountBondOption(kind, 1/coefficient, 2., 2.5)
            self.assertAlmostEqual(swaption_price(self.model, product, side), expected, delta=1e-10)
            self.assertEqual(ql.Settings.instance().evaluationDate, previous)

    def test_forward_mc_matches_native_engine_and_parity(self):
        values = monte_carlo_prices(self.model, self.product, paths=1 << 17)
        for side, (price, error) in values.items():
            self.assertAlmostEqual(price, swaption_price(self.model, self.product, side),
                                   delta=6*error+1e-7)
        curve = self.model.termStructure()
        swap = curve.discount(2)-curve.discount(7)-.02*sum(curve.discount(2+.5*j) for j in range(1,11))
        self.assertAlmostEqual(values['payer'][0]-values['receiver'][0], swap,
                               delta=6*(values['payer'][1]+values['receiver'][1])+1e-7)

    def test_invalid_diagnostic_configuration(self):
        with self.assertRaises(ValueError):
            swaption_price(self.model, self.product, 'call')
        with self.assertRaises(ValueError):
            monte_carlo_prices(self.model, self.product, paths=1)


if __name__ == '__main__':
    unittest.main()
