"""Guard row identity and conservative discrepancy reporting in the CIR probe."""
import copy
import unittest

from tools.performance.summarize_cir_forward_comparison import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.old = {'job': {'indices': [0, 900], 'paths': 1048576},
                    'rows': [{'source_index': 0, 'price': .1, 'standard_error': 1e-5},
                             {'source_index': 900, 'price': 0., 'standard_error': 0.}]}
        self.new = copy.deepcopy(self.old)

    def test_original_indices_survive_output_order(self):
        self.new['rows'].reverse()
        rows = compare(self.old, self.new)
        self.assertEqual([r['row_id'] for r in rows], ['000001', '000901'])
        self.assertEqual([r['regime'] for r in rows], ['core', 'stress'])

    def test_discrepancy_is_not_averaged_away(self):
        self.new['rows'][0]['price'] += .001
        rows = compare(self.old, self.new)
        self.assertFalse(rows[0]['screen_pass'])
        self.assertTrue(rows[1]['screen_pass'])

    def test_zero_sample_variance_has_no_z_score(self):
        row = compare(self.old, self.new)[1]
        self.assertIsNone(row['z_score'])
        self.assertIsNone(row['relative_difference'])
        self.assertEqual(row['screen_budget'], 2e-6)

    def test_missing_and_duplicate_rows_rejected(self):
        for rows in (self.new['rows'][:1], [self.new['rows'][0]]*2):
            with self.assertRaises(ValueError):
                compare(self.old, {**self.new, 'rows': rows})

    def test_different_workloads_rejected(self):
        for update in ({'side': 'receiver'}, {'paths': 65536}, {'indices': [0, 901]}):
            with self.assertRaises(ValueError):
                compare(self.old, {**self.new, 'job': {**self.new['job'], **update}})

    def test_nonfinite_rejected(self):
        self.new['rows'][0]['price'] = float('nan')
        with self.assertRaises(ValueError):
            compare(self.old, self.new)

    def test_reference_must_cover_every_row(self):
        with self.assertRaises(ValueError):
            compare(self.old, self.new, {0: {'fine_price': .1, 'mesh_difference': 0.}})
        pde = {index: {'fine_price': price, 'mesh_difference': 1e-7}
               for index, price in ((0, .1), (900, 1e-6))}
        self.assertTrue(all(r['forward_pde_screen_pass'] for r in compare(self.old, self.new, pde)))


if __name__ == '__main__':
    unittest.main()
