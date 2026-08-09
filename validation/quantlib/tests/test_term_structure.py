"""Tests for synthetic Actual/360 time-to-date conventions."""

from __future__ import annotations

import unittest

from validation.quantlib.term_structure import (
    REFERENCE_DATE,
    nearest_date_from_time,
)


class TermStructureTest(unittest.TestCase):
    def test_half_day_ties_match_cuda_nearest_step_rounding(self) -> None:
        maturity = 292.5 / 360.0

        maturity_date = nearest_date_from_time(maturity)

        self.assertEqual(maturity_date - REFERENCE_DATE, 293)


if __name__ == "__main__":
    unittest.main()
