"""CPU-only checks for the bounded exploratory LSM harness."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_fixed_income_lsm_probe as runner
from run_fixed_income_lsm_probe import bounded, cases
from summarize_lsm_probe import events, output_bytes


class ProbeToolsTest(unittest.TestCase):
    def test_power_excursions_do_not_interrupt_jobs_or_disappear_on_recovery(self):
        # Deliberately exceed the old 20% veto before, during and after jobs.
        def gpu_xml(power):
            return f"""<nvidia_smi_log><gpu><temperature><gpu_temp>90 C</gpu_temp></temperature>
                <gpu_power_readings><current_power_limit>{power} W</current_power_limit>
                <instant_power_draw>120 W</instant_power_draw></gpu_power_readings>
                <clocks><sm_clock>1500 MHz</sm_clock><mem_clock>9000 MHz</mem_clock></clocks>
                <fb_memory_usage><used>4000 MiB</used></fb_memory_usage>
                <utilization><gpu_util>100 %</gpu_util></utilization><performance_state>P0</performance_state>
                <clocks_event_reasons><clocks_event_reason_sw_thermal_slowdown>Active</clocks_event_reason_sw_thermal_slowdown>
                </clocks_event_reasons></gpu></nvidia_smi_log>"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "probe"
            binary.write_bytes(b"test binary")
            jobs = root / "jobs.json"
            jobs.write_text(json.dumps([{"id": name, "options": {"model": "cir"}}
                                       for name in ("first", "second")]))
            destination = root / "evidence"
            powers = iter((175, 130, 175, 175, 130, 175))
            launched = []

            def fake_bounded(command, stdout, stderr, timeout, monitor=None):
                if command[0] == "nvidia-smi":
                    stdout.write_text(gpu_xml(next(powers)))
                else:
                    launched.append(command)
                    self.assertIsNone(monitor(5))
                    stdout.write_text(json.dumps({"event": "summary", "gpu": {
                        "median_ms": 1, "coefficient_of_variation": 0}}))
                return {"returncode": 0, "timed_out": False, "stop_reason": None}

            with patch.object(sys, "argv", ["probe", "--jobs", str(jobs), "--binary", str(binary),
                                           "--output", str(destination)]), \
                 patch.object(runner, "bounded", side_effect=fake_bounded), \
                 patch.object(runner.subprocess, "run", return_value=SimpleNamespace(
                     returncode=0, stdout=gpu_xml(130))), \
                 patch.object(runner.subprocess, "check_output", side_effect=lambda *a, **kw:
                              "" if kw.get("text") else b""), \
                 patch("builtins.print"):
                self.assertEqual(runner.main(), 0)
            self.assertEqual(len(launched), 2)
            outcomes = [json.loads(line) for line in (destination / "results.ndjson").read_text().splitlines()]
            self.assertTrue(all(o["power_envelope_changed"] for o in outcomes))
            self.assertTrue(all(o["power_excursion_samples"] == 1 for o in outcomes))
            self.assertTrue(all(not o["timing_eligible"] for o in outcomes))
            self.assertTrue(all("GPU power limit changed by more than 20%" in o["timing_ineligibility_reasons"]
                                for o in outcomes))

    def test_other_models_keep_campaign_path_count(self):
        jobs = cases("other-models")
        self.assertEqual(len(jobs), 22)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        self.assertTrue(all(j["options"]["paths"] == 1 << 20 for j in jobs))

    def test_profiler_prefix_does_not_drop_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.ndjson"
            path.write_text('progress 50% {"event":"measurement","gpu_ms":1} suffix\n'
                            'ignored profiler output\n{"event":"summary"}\n')
            self.assertEqual([r["event"] for r in events(path)], ["measurement", "summary"])

    def test_fp32_comparison_includes_errors_and_signed_zero(self):
        original = {"prices": [1.0], "errors": [0.0]}
        self.assertNotEqual(output_bytes(original), output_bytes({"prices": [1.0], "errors": [-0.0]}))
        self.assertEqual(output_bytes(original), output_bytes(json.loads(json.dumps(original))))

    def test_watchdog_terminates_its_own_process(self):
        with tempfile.TemporaryDirectory() as directory:
            result = bounded([sys.executable, "-c", "import time; time.sleep(10)"],
                             Path(directory) / "out", Path(directory) / "err", 0.1)
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["returncode"], 0)
            self.assertLess(result["process_wall_seconds"], 5)

    def test_successful_process_is_not_marked_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = bounded([sys.executable, "-c", "print('ok')"],
                             Path(directory) / "out", Path(directory) / "err", 5)
            self.assertFalse(result["timed_out"])
            self.assertEqual(result["returncode"], 0)


if __name__ == "__main__":
    unittest.main()
