"""Tests for row-wise independent-reference fallback routing."""

from __future__ import annotations

from pathlib import Path
import unittest

from validation.hierarchy import (
    BackendBatchResult,
    BackendException,
    ValidationEngine,
    isolate_backend_exceptions,
    run_validation_hierarchy,
)


class ValidationHierarchyTest(unittest.TestCase):
    def test_only_backend_exceptions_descend_to_the_next_engine(self) -> None:
        requested_by_second: list[tuple[str, ...]] = []

        def premia(path, regime, row_ids):
            del path, regime
            return BackendBatchResult(
                completed_row_ids=row_ids[:2],
                exceptions=(
                    BackendException(row_ids[2], "domain failure", 12),
                    BackendException(row_ids[3], "non-finite result", 14),
                ),
                reports=("premia report",),
            )

        def quantlib(path, regime, row_ids):
            del path, regime
            requested_by_second.append(row_ids)
            return BackendBatchResult(
                completed_row_ids=row_ids[:1],
                exceptions=(BackendException(row_ids[1], "unsupported"),),
                reports=("quantlib report",),
            )

        hierarchy = run_validation_hierarchy(
            Path("prices.json"),
            "stress",
            ("000001", "000002", "000003", "000004"),
            (
                ValidationEngine("Premia", "specialized pricer", premia),
                ValidationEngine("QuantLib", "specialized pricer", quantlib),
            ),
        )

        self.assertEqual(requested_by_second, [("000003", "000004")])
        self.assertEqual(hierarchy.unresolved_row_ids, ("000004",))
        self.assertEqual(hierarchy.primary_reference, "Premia (specialized pricer)")

    def test_no_compatible_engine_leaves_every_row_unresolved(self) -> None:
        hierarchy = run_validation_hierarchy(
            "prices.json", "core", ("000001", "000002"), ()
        )

        self.assertEqual(hierarchy.runs, ())
        self.assertEqual(hierarchy.unresolved_row_ids, ("000001", "000002"))
        self.assertEqual(hierarchy.primary_reference, "none")

    def test_unavailable_engine_is_recorded_without_being_called(self) -> None:
        engine = ValidationEngine(
            "QuantLib",
            "specialized pricer",
            None,
            "no compatible engine",
        )

        hierarchy = run_validation_hierarchy(
            "prices.json", "core", ("000001",), (engine,)
        )

        self.assertEqual(hierarchy.engine_plan, (engine,))
        self.assertEqual(hierarchy.runs, ())
        self.assertEqual(hierarchy.unresolved_row_ids, ("000001",))

    def test_completed_comparison_failures_do_not_fall_through(self) -> None:
        second_engine_called = False

        def first(path, regime, row_ids):
            del path, regime
            return BackendBatchResult(row_ids, (), ("comparison failure",))

        def second(path, regime, row_ids):
            nonlocal second_engine_called
            del path, regime, row_ids
            second_engine_called = True
            raise AssertionError("A finite comparison must not fall through.")

        hierarchy = run_validation_hierarchy(
            "prices.json",
            "core",
            ("000001",),
            (
                ValidationEngine("Premia", "specialized pricer", first),
                ValidationEngine("QuantLib", "specialized pricer", second),
            ),
        )

        self.assertFalse(second_engine_called)
        self.assertEqual(hierarchy.unresolved_row_ids, ())

    def test_exception_isolation_preserves_successful_batches(self) -> None:
        def validate_batch(row_ids):
            if "000003" in row_ids:
                raise RuntimeError("backend failed")
            return row_ids

        result = isolate_backend_exceptions(
            ("000001", "000002", "000003", "000004"),
            validate_batch,
            lambda error: isinstance(error, RuntimeError),
        )

        self.assertEqual(
            set(result.completed_row_ids), {"000001", "000002", "000004"}
        )
        self.assertEqual(
            tuple(exception.row_id for exception in result.exceptions),
            ("000003",),
        )


if __name__ == "__main__":
    unittest.main()
