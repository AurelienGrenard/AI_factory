"""Validate the permanent price-gradient performance strategy matrix."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from tools.performance.run_price_gradient_strategies import (
    load_manifest,
    parse_ncu_raw_csv,
    representative_diagnostics,
    sass_local_instructions,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "tests" / "performance" / "price_gradients"
    / "strategy_manifest.json"
)


class PriceGradientStrategyManifestTest(unittest.TestCase):
    def test_manifest_covers_every_active_strategy(self) -> None:
        document = load_manifest(MANIFEST)
        case_ids = {case["id"] for case in document["cases"]}
        self.assertEqual(
            case_ids,
            {
                "terminal_mc_exact_and_step_b1_256",
                "closed_form_thread_256",
                "heston_american_frozen_first_lsm_b1_256",
                "heston_american_frozen_diagonal_lsm_b1_256",
                "cir_jamshidian_first_scalar_and_cooperative",
                "cir_jamshidian_diagonal_scalar_and_cooperative",
            },
        )
        kernels = {
            kernel
            for case in document["cases"]
            for kernel in case["required_kernels"]
        }
        for phase in (
            "prepare_rows",
            "simulate_paths",
            "regression_partials",
            "solve_regressions",
            "update_cashflows",
            "moment_partials",
            "finalize_prices",
            "frozen_sensitivity_moments",
            "finalize_frozen_sensitivities",
        ):
            self.assertIn(
                f"heston.american_option.sensitivities.{phase}",
                kernels,
            )

    def test_commands_and_kernel_names_are_unique(self) -> None:
        document = json.loads(MANIFEST.read_text())
        commands = []
        for case in document["cases"]:
            command = (case["binary"], *case["arguments"])
            self.assertNotIn(command, commands)
            commands.append(command)
            self.assertEqual(
                len(case["required_kernels"]),
                len(set(case["required_kernels"])),
            )

    def test_sass_counts_only_launched_symbols(self) -> None:
        sass = """Function : launched
        /*0000*/ LDL R0, [R1];
        /*0010*/ STL [R1], R0;
Function : compiled_but_unused
        /*0000*/ LDL R0, [R1];
"""
        self.assertEqual(
            sass_local_instructions(sass, {"launched"}),
            {"launched": 2},
        )

    def test_profile_representative_is_heaviest_required_launch(self) -> None:
        def diagnostic(kernel: str, registers: int, blocks: int) -> dict:
            return {
                "kernel": kernel,
                "compiled_symbol": f"{kernel}-{registers}-{blocks}",
                "resources": {
                    "registers_per_thread": registers,
                    "local_bytes_per_thread": 0,
                },
                "launch": {
                    "dynamic_shared_bytes_per_block": 0,
                    "grid_block_count": blocks,
                },
            }

        selected = representative_diagnostics(
            ["phase_a", "phase_b"],
            [
                diagnostic("phase_a", 32, 100),
                diagnostic("phase_a", 40, 1),
                diagnostic("phase_b", 24, 4),
            ],
        )
        self.assertEqual(
            [record["compiled_symbol"] for record in selected],
            ["phase_a-40-1", "phase_b-24-4"],
        )

    def test_ncu_summary_keeps_resource_and_stall_evidence(self) -> None:
        parsed = parse_ncu_raw_csv(
            '"Grid Size","Block Size","gpu__time_duration.avg",'
            '"derived__local_spilling_requests",'
            '"smsp__average_warps_issue_stalled_wait_per_issue_active.ratio"\n'
            '"","","us","request","warp"\n'
            '"(4, 1, 1)","(256, 1, 1)","12.5","0","2.25"\n'
        )
        self.assertEqual(parsed["grid_size"], "(4, 1, 1)")
        self.assertEqual(parsed["duration_us"], 12.5)
        self.assertEqual(parsed["local_spilling_requests"], 0.0)
        self.assertEqual(
            parsed["leading_stalls"],
            [{"reason": "wait", "ratio": 2.25}],
        )


if __name__ == "__main__":
    unittest.main()
