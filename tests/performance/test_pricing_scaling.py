"""Host-only tests for capability-derived scaling coverage and fail-closed evidence."""
import json
import io
import copy
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.performance.pricing_scaling_manifest import (
    MODEL_SPECS, ROOT, coverage, generate, render_case, workloads,
)
from tools.performance.run_pricing_scaling import (
    compiled_thread_limit, digest, jobs_for, production_jobs, read_job_progress,
    run_process,
)
from tools.performance.experiment_environment import (
    hardware_power_brake_issue, power_comparability_issue, record_timing_issue,
    validate_experiment_preflight,
)
from tools.performance.run_baseline import validate_preflight
from tools.performance.pricing_scaling_inputs import (
    CATALOGUE_INPUT_PROFILE, TILE_INDICES, expand_document, prepare_inputs,
    source_indices,
)
from tools.performance.plan_pricing_scaling_confirmation import (
    confirmation_jobs, execution_key,
)
from tools.performance.summarize_pricing_scaling import (
    geometry_envelope, single_price_information, summarize, within_tolerance,
)
from tools.performance.export_pricing_dataset_runtime import (
    case_result, export_campaign, timing_breakdown,
)


class PricingScalingTest(unittest.TestCase):
    def test_production_jobs_use_the_compiled_plan_without_replacing_paths(self):
        case = next(c for c in workloads() if c["id"] == "heston__mc_terminal")
        plan = {"price_count": 1000, "paths_per_price": 1 << 20,
                "threads_per_block": 128, "prices_per_launch": 1000, "block_count": 256}
        module = production_jobs.__module__
        with patch(module + ".subprocess.run", return_value=Mock(stdout=json.dumps(plan))) as run, \
                patch(module + ".digest", return_value="inspector-hash"):
            jobs, evidence = production_jobs(case, Path("/tmp/test-build"), [1000])
        self.assertEqual(jobs[0]["threads"], 128)
        self.assertEqual(jobs[0]["block_limit"], 256)
        self.assertEqual(jobs[0]["paths"], 1 << 20)
        self.assertEqual(evidence[0]["inspector_sha256"], "inspector-hash")
        self.assertEqual(run.call_args.args[0][1:], ["heston/european_option", "1000"])
        plan["paths_per_price"] = 65536
        with patch(module + ".subprocess.run", return_value=Mock(stdout=json.dumps(plan))):
            with self.assertRaises(ValueError):
                production_jobs(case, Path("/tmp/test-build"), [1000])

    def test_every_model_is_classified_and_lsm_is_not_executed(self):
        cases = workloads()
        self.assertEqual({c["model"] for c in cases}, {m.name for m in MODEL_SPECS})
        self.assertEqual(len(cases), len({c["id"] for c in cases}))
        self.assertTrue(all("lsm" not in c["engine"] for c in cases))
        self.assertEqual(len(coverage(cases)), len(MODEL_SPECS))

    def test_every_available_configuration_has_one_representative(self):
        cases = workloads()
        counts = {}
        for case in cases:
            counts[case["model"], case["family"]] = (
                counts.get((case["model"], case["family"]), 0) + 1
            )
        for model in MODEL_SPECS:
            if model.asset_class == "equity":
                self.assertEqual(counts.get((model.name, "mc_barrier")), 1)
                self.assertEqual(
                    counts.get((model.name, "mc_terminal"), 0),
                    0 if model.name == "black_scholes" else 1,
                )
                self.assertEqual(
                    counts.get((model.name, "closed_form"), 0),
                    1 if model.name == "black_scholes" else 0,
                )
            else:
                self.assertGreaterEqual(counts.get((model.name, "closed_form"), 0), 1)
                self.assertEqual(counts.get((model.name, "mc_terminal"), 0), 0)
                self.assertEqual(counts.get((model.name, "mc_barrier"), 0), 0)

    def test_only_one_side_and_product_per_configuration(self):
        for case in workloads():
            expected = {"mc_terminal": ("european_option", "call"),
                        "mc_barrier": ("down_and_out_option", "put"),
                        "closed_form": ("european_option", "call") if case["model"] == "black_scholes"
                        else ("rate_option", "call")}[case["family"]]
            self.assertEqual((case["product"], case["side"]), expected)
            self.assertTrue((ROOT / case["header"]).is_file())
            self.assertTrue((ROOT / case["generator"]).is_file())

    def test_generated_adapters_have_no_unsubstituted_field(self):
        for case in workloads(include_lsm=True):
            source = render_case(case)
            self.assertNotIn("$", source)
            self.assertIn("launch_", source)
            self.assertNotIn("__global__", source)
            closed_form = str(case["family"] == "closed_form").lower()
            self.assertIn(f"read_jobs(argc, argv, {closed_form})", source)

    def test_lsm_coverage_and_aligned_slice_seed_mapping(self):
        cases = [c for c in workloads(include_lsm=True) if c["family"] == "lsm"]
        self.assertEqual(len(cases), 19)
        self.assertEqual(sum(c["product"] == "american_option" for c in cases), 9)
        for case in cases:
            source = render_case(case)
            self.assertIn("return model_binding::", source)
            self.assertIn("seed + offset", source)
            self.assertIn("device_models.template as<Model>() + offset", source)
            self.assertIn("host_products", (ROOT / case["header"]).read_text())
            self.assertEqual(len(jobs_for(case, "scaling")), 9)
            geometries = jobs_for(case, "geometry")
            self.assertEqual(len({j["id"] for j in geometries}), len(geometries))
            self.assertEqual({j["blocks_per_price"] for j in geometries}, {32, 64, 128})
            self.assertEqual({j["threads"] for j in geometries}, {128, 256})
            self.assertEqual({j["batch_rows"] for j in geometries if j["rows"] == 1000},
                             {100, 1000})

    def test_lsm_measurements_use_native_gpu_events_and_regression_diagnostics(self):
        source = (ROOT / "tests/performance/pricing_scaling_support.cuh").read_text()
        self.assertIn("validate_regression_diagnostics(result, case_id)", source)
        self.assertIn("lsm_gpu_ms += result.kernel_seconds * 1000.0", source)
        self.assertIn("device_memory_json(transient_peak_bytes)", source)

    def test_lsm_binding_only_scope_and_exact_calendar_are_explicit(self):
        cases = {c["id"]: c for c in workloads(include_lsm=True)}
        black_scholes = cases["black_scholes__lsm"]
        self.assertIsNone(black_scholes["generator"])
        self.assertEqual(black_scholes["scope"], "binding_only_no_price_recipe")
        self.assertEqual(black_scholes["seed_source"], "scaling_only_sha256_case_id_56bit")
        self.assertLess(black_scholes["seed"] + 1000, 2**64)
        self.assertEqual(black_scholes["time_kind"], "exact")
        self.assertEqual(cases["cir__lsm"]["time_kind"], "exact")
        for curve in ("nelson_siegel", "svensson"):
            self.assertEqual(cases[f"cir_plus_plus__{curve}__lsm"]["time_kind"], "exact")
        self.assertEqual(cases["vasicek__lsm"]["time_kind"], "exact")
        self.assertIsNone(cases["black_scholes__closed_form"]["time_kind"])
        self.assertNotIn('"1 / 504"', render_case(black_scholes))
        self.assertIn('"1 / 504"', render_case(cases["heston__lsm"]))

    def test_no_mathdx_preserves_all_independent_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = generate(Path(directory), False)
            self.assertEqual(set(manifest["not_built"]),
                             {c["id"] for c in workloads() if c["mathdx"]})

    def test_grid_means_one_million_paths_per_price(self):
        case = next(c for c in workloads() if c["family"] == "mc_terminal")
        jobs = jobs_for(case, "scaling")
        self.assertEqual(len(jobs), 9)
        self.assertEqual({j["rows"] for j in jobs}, {100, 1000, 10000})
        self.assertTrue(any(j["rows"] == 10000 and j["paths"] == 1048576 for j in jobs))
        self.assertEqual(max(j["rows"] * j["paths"] for j in jobs), 10485760000)

    def test_closed_form_does_not_multiply_path_counts(self):
        case = next(c for c in workloads() if c["family"] == "closed_form")
        self.assertEqual(len(jobs_for(case, "scaling")), 3)

    def test_mc_geometry_covers_threads_grid_and_batches(self):
        case = next(c for c in workloads()
                    if c["family"] == "mc_terminal" and not c["mathdx"])
        jobs = jobs_for(case, "geometry", sm_count=76)
        thousand = [j for j in jobs if j["rows"] == 1000]
        self.assertEqual({j["threads"] for j in thousand}, {128, 256, 512})
        self.assertTrue({152, 1000}.issubset(
            {j["block_limit"] for j in thousand}
        ))
        self.assertTrue({100, 1000}.issubset(
            {j["batch_rows"] for j in thousand}
        ))
        self.assertTrue(any(j["rows"] == 10000 and j["batch_rows"] == 4096
                            and j["block_limit"] == 4096 for j in jobs))
        for template in ("markovian", "rough/markovian_n_factor"):
            source = (ROOT / "tools/codegen/pricing_bindings/templates/catalog/pricing"
                      / template / "generator.cpp.tpl").read_text()
            self.assertIn("cuda_tuning::kMonteCarloRowsPerLaunch", source)
            self.assertIn("cuda_tuning::kMonteCarloBlockCountLimit", source)

    def test_geometry_candidates_cross_both_workload_axes(self):
        for family in ("mc_terminal", "mc_barrier", "lsm"):
            case = next(c for c in workloads(include_lsm=True) if c["family"] == family)
            jobs = jobs_for(case, "geometry")
            self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
            for rows in (100, 1000, 10000):
                for paths in (65536, 262144, 1048576):
                    point = [j for j in jobs if j["rows"] == rows and j["paths"] == paths]
                    self.assertGreaterEqual(len(point), 2)
                    self.assertTrue(all(j["warmups"] >= 1 and j["repetitions"] >= 3 for j in point))

    def test_compiled_ceiling_bounds_mc_candidates_without_changing_the_kernel(self):
        case = next(c for c in workloads() if c["id"] == "rough_heston__mc_terminal")
        jobs = jobs_for(case, "screening", price_counts=(100,), path_counts=(65536,),
                        thread_limit=384)
        self.assertEqual({j["threads"] for j in jobs}, {64, 128, 256, 384})
        self.assertEqual(len(jobs), 4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "calibration" / case["id"]
            evidence.mkdir(parents=True)
            binary = root / ("ai_factory_pricing_scaling_" + case["id"])
            binary.write_bytes(b"test-only binary identity")
            (evidence / "provenance.json").write_text(json.dumps({"binary_sha256": digest(binary)}))
            (evidence / "outcome.json").write_text(json.dumps({"status": "measured"}))
            (evidence / "resources.json").write_text(json.dumps([{
                "compiled_symbol": "test-only symbol", "device": {"name": "test GPU"},
                "resources": {"maximum_threads_per_block": 384}}]))
            limit = compiled_thread_limit(case, root, root / "calibration")
            self.assertEqual(limit["maximum_threads"], 384)
            self.assertEqual(len(limit["source_hashes"]), 3)
            binary.write_bytes(b"different compilation")
            with self.assertRaises(ValueError):
                compiled_thread_limit(case, root, root / "calibration")

    def test_stratified_repeated_tile_has_identical_composition_at_every_scale(self):
        self.assertEqual(sum(i < 900 for i in TILE_INDICES), 90)
        for count in (100, 1000, 10000):
            indices = source_indices(count)
            self.assertEqual(sum(i >= 900 for i in indices), count // 10)
            self.assertEqual(indices[:100], list(TILE_INDICES))
            self.assertEqual({i: indices.count(i) for i in TILE_INDICES},
                             {i: count // 100 for i in TILE_INDICES})
        for invalid in (0, 10001):
            with self.assertRaises(ValueError):
                source_indices(invalid)

    def test_confirmation_preserves_per_shape_candidates_and_native_batch_reference(self):
        case = next(c for c in workloads() if c["id"] == "kou__mc_terminal")
        source = [{"id": str(p), "rows": p, "paths": 1048576,
                   "threads": t, "batch_rows": p, "block_limit": p,
                   "path_chunk": 65536, "publication_directory": "/must-not-reuse"}
                  for p, t in ((100, 1024), (1000, 512), (10000, 64))]
        original = json.dumps(source)
        points = [{"rows": j["rows"], "paths_per_price": j["paths"],
                   "candidate_id": j["id"], "gpu_median_ms": 100} for j in source]
        jobs, decisions = confirmation_jobs(case, points, source, 1000, 7, 3)
        self.assertEqual(len(jobs), 5)  # equivalent 1000-row native grid caps are not duplicated
        candidates = [j for j in jobs if j["id"].endswith("candidate")]
        self.assertEqual([j["threads"] for j in candidates], [1024, 512, 64])
        self.assertTrue(any(j["rows"] == 10000 and j["batch_rows"] == 4096
                            and j["block_limit"] == 4096 for j in jobs))
        self.assertTrue(all(j["operations_per_sample"] == 10 for j in jobs))
        self.assertTrue(all("publication_directory" not in j for j in jobs))
        self.assertTrue(all(not d["production_tuning_accepted"] for d in decisions))
        self.assertEqual(json.dumps(source), original)
        self.assertEqual(execution_key(source[1]), execution_key({**source[1], "block_limit": 4096}))

    def test_temporary_inputs_keep_sources_immutable_and_rows_aligned(self):
        document = {"database_id": "example", "row_count": 1000,
                    "models": [{"id": str(i), "parameters": {"value": i}} for i in range(1000)]}
        original = json.dumps(document)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(original)
            result = prepare_inputs({"model": source}, root / "fixture", [100, 1000, 10000])
            self.assertEqual(source.read_text(), original)
            expanded = json.loads(Path(result["by_count"]["10000"]["model"]).read_text())
            self.assertEqual(len({r["id"] for r in expanded["models"]}), 10000)
            self.assertEqual(expanded["models"][100]["parameters"], expanded["models"][0]["parameters"])
            self.assertTrue(expanded["benchmark_only"])
            self.assertEqual(json.dumps(document), original)

    def test_geometry_envelope_allows_a_different_configuration_at_each_shape(self):
        rows = []
        for prices, threads, milliseconds in ((100, 512, 1), (100, 128, 2),
                                              (1000, 512, 12), (1000, 128, 10)):
            rows.append({"id": f"p{prices}_t{threads}", "run": "confirmed",
                "outcome": "measured", "timing_eligible": True, "environment_policy": "strict",
                "fixture_signature": "same-input-tile", "binary_sha256": "same-binary",
                "configuration": {"rows": prices, "paths_per_price": 65536,
                                  "requested_threads": threads, "warmups": 1, "repetitions": 3},
                "gpu": {"median_ms": milliseconds, "coefficient_of_variation": .01}})
        envelope, comparisons = geometry_envelope(rows)
        self.assertEqual([p["configuration"]["requested_threads"] for p in envelope], [512, 128])
        self.assertEqual(comparisons[0]["axis"], "rows")
        self.assertEqual(comparisons[0]["normalized_cost_ratio"], 1)
        self.assertEqual(comparisons[0]["assessment"], "no_superlinear_alert_on_this_pair")
        self.assertTrue(all(not p["production_tuning_accepted"] for p in envelope))
        rows[-1]["fixture_signature"] = "different-inputs"
        self.assertEqual(geometry_envelope(rows)[1][0]["assessment"], "inconclusive")
        rows[-1]["fixture_signature"] = "same-input-tile"
        rows[-1]["run"] = "different-operating-period"
        self.assertEqual(geometry_envelope(rows)[1][0]["assessment"], "inconclusive")

    def test_catalogue_profile_preserves_every_ordered_row_and_has_distinct_provenance(self):
        document = {"database_id": "example", "row_count": 1000,
                    "models": [{"id": f"original-{i}", "parameters": {"value": i}}
                               for i in range(1000)]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(document))
            before = source.read_bytes()
            native = prepare_inputs({"model": source}, root / "native", [1000], CATALOGUE_INPUT_PROFILE)
            tile = prepare_inputs({"model": source}, root / "tile", [1000])
            copied = json.loads(Path(native["by_count"]["1000"]["model"]).read_text())
            self.assertEqual(copied["models"], document["models"])
            self.assertEqual(native["source_indices_zero_based"], list(range(1000)))
            self.assertNotEqual(native["fixture_signature"], tile["fixture_signature"])
            self.assertEqual(source.read_bytes(), before)
            for rows in (100, 10000):
                with self.assertRaises(ValueError):
                    prepare_inputs({"model": source}, root / f"invalid-{rows}", [rows], CATALOGUE_INPUT_PROFILE)

    def test_missing_measurements_never_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "plan.json").write_text(json.dumps({"manifest": {"cases": workloads()}, "stage": "scaling"}))
            result = summarize([path])
            self.assertEqual(result["missing_case_count"], len(workloads()))

    def test_numerical_conflict_excludes_both_geometries_despite_stable_timings(self):
        case = next(c for c in workloads() if c["family"] == "mc_terminal")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            case_path = path / case["id"]
            case_path.mkdir()
            (path / "plan.json").write_text(json.dumps({"manifest": {"cases": [case]},
                "stage": "geometry", "environment_policy": "strict"}))
            (case_path / "provenance.json").write_text(json.dumps({
                "fixture": {"fixture_signature": "same"}, "binary_sha256": "same"}))
            records = []
            for name, paths, price, milliseconds in (("small", 262144, 1, 10),
                    ("reference", 1048576, 1, 40), ("candidate", 1048576, 1.001, 30)):
                records.append({"event": "scaling_result", "case": case["id"], "id": name,
                    "configuration": {"rows": 100, "offset": 0, "paths_per_price": paths,
                        "seed": 0, "warmups": 1, "repetitions": 3},
                    "gpu": {"median_ms": milliseconds, "coefficient_of_variation": .01},
                    "prices": [price] * 100, "standard_errors": [.01] * 100,
                    "public_api": {}, "preparation_once_ms": 0,
                    "publication": None, "device_memory": {}})
            (case_path / "stdout.ndjson").write_text("\n".join(json.dumps(r) for r in records))
            (case_path / "outcome.json").write_text(json.dumps({
                "status": "measured", "timing_eligible": True}))
            result = summarize([path])
            self.assertEqual(result["numerical_failures"], 1)
            report = result["cases"][0]
            self.assertEqual([r["numerical_conflict"] for r in report["measurements"]], [False, True, True])
            self.assertFalse(report["geometry_envelope"][1]["repeated_timing_qualified"])
            self.assertIn((100, 1048576), report["unqualified_shapes"])
            self.assertEqual(report["comparisons"][0]["assessment"], "inconclusive")

    def test_geometry_envelope_keeps_distinct_row_offsets_separate(self):
        rows = [{"id": str(offset), "run": "same", "outcome": "measured",
                 "timing_eligible": False, "environment_policy": "observe",
                 "configuration": {"rows": 1, "offset": offset, "paths_per_price": 1048576},
                 "gpu": {"median_ms": 1}} for offset in (37, 57)]
        envelope, comparisons = geometry_envelope(rows)
        self.assertEqual([r["offset"] for r in envelope], [37, 57])
        self.assertEqual(comparisons, [])

    def test_single_price_ratio_is_information_not_a_scaling_verdict(self):
        points = [{"rows": prices, "paths_per_price": 65536,
                   "gpu_median_ms": milliseconds, "run": "example"}
                  for prices, milliseconds in ((1, 2), (100, 10), (1000, 100))]
        ratios = single_price_information(points)
        self.assertEqual([r["gpu_time_over_prices_times_single_price"] for r in ratios], [.05, .05])
        self.assertTrue(all(r["assessment"] == "informational_only" for r in ratios))
        rows = [{"id": str(p["rows"]), "run": "example", "outcome": "measured",
                 "timing_eligible": True, "environment_policy": "strict",
                 "fixture_signature": "same", "binary_sha256": "same",
                 "configuration": {"rows": p["rows"], "paths_per_price": 65536,
                                   "warmups": 1, "repetitions": 3},
                 "gpu": {"median_ms": p["gpu_median_ms"], "coefficient_of_variation": .01}}
                for p in points]
        comparisons = geometry_envelope(rows)[1]
        self.assertEqual(len(comparisons), 1)
        self.assertEqual((comparisons[0]["from"], comparisons[0]["to"]), (100, 1000))

    def test_summary_keeps_each_timing_boundary(self):
        case = next(c for c in workloads() if c["family"] == "closed_form")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            case_path = path / case["id"]
            case_path.mkdir()
            (path / "plan.json").write_text(json.dumps({
                "manifest": {"cases": [case]}, "stage": "geometry"
            }))
            record = {
                "event": "scaling_result", "case": case["id"], "id": "boundary",
                "configuration": {"rows": 1000, "offset": 0, "paths_per_price": 0,
                                  "requested_threads": 256, "batch_rows": 1000,
                                  "block_limit": 4096, "path_chunk": 0, "seed": 0,
                                  "repetitions": 3},
                "gpu": {"median_ms": .01, "coefficient_of_variation": .01},
                "public_api": {"median_ms": .02, "coefficient_of_variation": .01},
                "raw_host_clock": {"median_ms": .007, "coefficient_of_variation": .01},
                "raw_host_samples_ms": [.006, .007, .008],
                "preparation_once_ms": 3.0, "operations_per_sample": 1024,
                "output_copy": {"median_ms": .04},
                "publication": {"wall_ms": 5.0}, "device_memory": {"tracked_peak_bytes": 8},
                "prices": [1.0], "standard_errors": [], "timing_scope": "explicit",
            }
            (case_path / "stdout.ndjson").write_text(json.dumps(record) + "\n")
            (case_path / "outcome.json").write_text(json.dumps({"status": "measured"}))
            measurement = summarize([path])["cases"][0]["measurements"][0]
            self.assertEqual(measurement["operations_per_sample"], 1024)
            self.assertEqual(measurement["output_copy"]["median_ms"], .04)
            self.assertEqual(measurement["publication"]["wall_ms"], 5.0)
            self.assertEqual(measurement["timing_scope"], "explicit")
            self.assertEqual(measurement["raw_host_clock"]["median_ms"], .007)
            self.assertEqual(measurement["raw_host_samples_ms"], [.006, .007, .008])
            # An old capture lacks the independent clock; do not substitute
            # its clamped public-API interval or invent historical samples.
            del record["raw_host_clock"]
            del record["raw_host_samples_ms"]
            (case_path / "stdout.ndjson").write_text(json.dumps(record) + "\n")
            legacy = summarize([path])["cases"][0]["measurements"][0]
            self.assertIsNone(legacy["raw_host_clock"])
            self.assertIsNone(legacy["raw_host_samples_ms"])

    def test_numeric_failure_nonfinite_and_length_are_rejected(self):
        self.assertTrue(within_tolerance([1], [1 + 1e-7], 1e-6, 1e-6))
        self.assertFalse(within_tolerance([1], [2], 1e-6, 1e-6))
        self.assertFalse(within_tolerance([1], [], 1e-6, 1e-6))
        self.assertFalse(within_tolerance([float("nan")], [1], 1e-6, 1e-6))

    def test_power_excursion_excludes_timings_even_at_a_process_boundary(self):
        self.assertIsNone(power_comparability_issue(175, 175, 140, 175))
        outcome = {"timing_eligible": True}
        record_timing_issue(outcome, power_comparability_issue(150, 175, 140, 175))
        record_timing_issue(outcome, power_comparability_issue(175, 175, 140, 175))
        self.assertFalse(outcome["timing_eligible"])
        self.assertEqual(len(outcome["timing_ineligibility_reasons"]), 1)
        for value in (None, 0, float("nan"), 55, 180):
            self.assertIsNotNone(power_comparability_issue(value, 175, 140, 175))

    def test_experiment_preflight_does_not_mutate_or_weaken_official_admission(self):
        profile = json.loads((ROOT / "tests/performance/baseline_sm89_v3.json").read_text())
        original = copy.deepcopy(profile)
        snapshot = {"gpu": profile["environment"]["gpu"], "power_source": "external_power",
                    "power_limits_w": {"current": 55}, "concurrent_compute_processes": [],
                    "throttle": {"hardware_power_brake": "Not Active"}}
        validate_experiment_preflight(profile, snapshot)
        self.assertEqual(profile, original)
        with self.assertRaises(ValueError):
            validate_preflight(profile, snapshot)
        for changed in ({"power_source": "battery"}, {"concurrent_compute_processes": [123]},
                        {"throttle": {"hardware_power_brake": "Active"}}):
            with self.assertRaises(ValueError):
                validate_experiment_preflight(profile, {**snapshot, **changed})

    def test_running_process_survives_power_excursion_and_recovery(self):
        module = run_process.__module__
        process = Mock(pid=123, returncode=0)
        process.poll.side_effect = [None, None, 0, 0]
        process.wait.side_effect = [subprocess.TimeoutExpired("probe", 5)] * 2
        sample = {"temperature_c": 85, "power_limit_w": 150,
                  "throttle": {"clocks_event_reason_sw_thermal_slowdown": "Active"},
                  "compute_process_ids": [123], "utilization_percent": 100, "sm_clock_mhz": 1590}
        with tempfile.TemporaryDirectory() as directory, \
             patch(module + ".subprocess.Popen", return_value=process), \
             patch(module + ".telemetry", side_effect=[("<gpu/>", dict(sample)),
                   ("<gpu/>", {**sample, "power_limit_w": 175})]), \
             patch(module + ".time.monotonic", side_effect=[0, 5, 10, 11]), \
             patch(module + ".os.killpg") as kill:
            result = run_process(["probe"], Path(directory), 60, 175,
                                 "strict", 140, 175)
            self.assertEqual(result["returncode"], 0)
            self.assertIsNone(result["stop_reason"])
            self.assertFalse(result["timing_eligible"])
            self.assertEqual(len(result["timing_ineligibility_reasons"]), 1)
            kill.assert_not_called()
            rows = [json.loads(line) for line in
                    (Path(directory) / "telemetry.ndjson").read_text().splitlines()]
            self.assertIsNotNone(rows[0]["timing_comparability_issue"])
            self.assertIsNone(rows[1]["timing_comparability_issue"])

    def test_running_process_still_stops_on_hardware_power_brake(self):
        module = run_process.__module__
        process = Mock(pid=123, returncode=-15)
        process.poll.side_effect = [None, None]
        process.wait.side_effect = [subprocess.TimeoutExpired("probe", 5), -15]
        sample = {"power_limit_w": 175,
                  "throttle": {"clocks_event_reason_hw_power_brake_slowdown": "Active"},
                  "compute_process_ids": [123], "utilization_percent": 100, "sm_clock_mhz": 1590}
        with tempfile.TemporaryDirectory() as directory, \
             patch(module + ".subprocess.Popen", return_value=process), \
             patch(module + ".telemetry", return_value=("<gpu/>", sample)), \
             patch(module + ".time.monotonic", side_effect=[0, 5, 6]), \
             patch(module + ".os.killpg") as kill:
            result = run_process(["probe"], Path(directory), 60, 175)
            self.assertEqual(result["stop_reason"], "GPU hardware power brake")
            self.assertFalse(result["timing_eligible"])
            kill.assert_called_once()

    def test_progress_association_ignores_results_and_preserves_partial_records(self):
        first = '{"event":"job_start","id":"a","rows":100,"paths":65536}\n'
        stream = io.StringIO(first + '{"event":"scaling_result","prices":[]}\n'
                             + '{"event":"job_start","id":"unfinished')
        progress = read_job_progress(stream, None)
        self.assertEqual(progress, {"id": "a", "rows": 100, "paths": 65536})
        self.assertTrue(stream.read().startswith('{"event":"job_start","id":"unfinished'))

    def test_native_observation_records_temperature_without_a_veto(self):
        for reason in ("sw_thermal_slowdown", "hw_thermal_slowdown", "hw_slowdown"):
            self.assertIsNone(hardware_power_brake_issue({"clocks_event_reason_" + reason: "Active"}))
        self.assertIsNotNone(hardware_power_brake_issue(
            {"clocks_event_reason_hw_power_brake_slowdown": "Active"}))

    def test_power_excluded_scaling_cannot_qualify_even_with_low_noise(self):
        case = next(c for c in workloads() if c["family"] == "mc_terminal")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            case_path = path / case["id"]
            case_path.mkdir()
            (path / "plan.json").write_text(json.dumps({"manifest": {"cases": [case]},
                "stage": "scaling", "environment_policy": "strict"}))
            records = []
            for paths in (65536, 1048576):
                records.append({"event": "scaling_result", "case": case["id"], "id": str(paths),
                    "configuration": {"rows": 1000, "offset": 0, "paths_per_price": paths,
                        "requested_threads": 512, "batch_rows": 1000, "block_limit": 4096,
                        "path_chunk": 0, "seed": 0, "warmups": 1, "repetitions": 3},
                    "gpu": {"median_ms": paths / 65536, "coefficient_of_variation": .01},
                    "public_api": {}, "preparation_once_ms": 0, "publication": None,
                    "device_memory": {}, "prices": [], "standard_errors": []})
            (case_path / "stdout.ndjson").write_text("\n".join(json.dumps(r) for r in records))
            (case_path / "outcome.json").write_text(json.dumps({
                "status": "measured", "timing_eligible": False,
                "timing_ineligibility_reasons": ["GPU power limit changed by more than 10%"]}))
            result = summarize([path])["cases"][0]
            self.assertEqual(result["comparisons"][0]["assessment"], "inconclusive")
            self.assertTrue(all(not r["timing_eligible"] for r in result["measurements"]))
            self.assertTrue(all(r["timing_ineligibility_reasons"] for r in result["measurements"]))
            # Observation mode cannot qualify even if a producer reports eligibility.
            plan = json.loads((path / "plan.json").read_text())
            plan["environment_policy"] = "observe"
            (path / "plan.json").write_text(json.dumps(plan))
            (case_path / "outcome.json").write_text(json.dumps({
                "status": "measured", "timing_eligible": True}))
            self.assertEqual(summarize([path])["cases"][0]["comparisons"][0]["assessment"], "inconclusive")

    def test_dataset_runtime_phase_sum_never_counts_gpu_twice(self):
        row = {"preparation_once_ms": 10, "raw_host_clock": {"median_ms": 100},
               "output_copy": {"median_ms": 2}, "publication": {"wall_ms": 3, "native_writer": True},
               "gpu": {"median_ms": 120}, "public_api": {"median_ms": 120}}
        result = timing_breakdown(row)
        self.assertEqual(result["generation_phase_sum_ms"], 115)
        self.assertEqual(result["gpu_median_ms"], 120)
        row["preparation_once_ms"] = float("nan")
        with self.assertRaises(ValueError):
            timing_breakdown(row)

    def test_dataset_runtime_export_refuses_tile_and_wrong_target_workloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text(json.dumps({"input_profile": "repeated_stratified_90_core_10_stress_v1"}))
            with self.assertRaises(ValueError):
                export_campaign(root)
            case = {"id": "heston__mc_terminal", "family": "mc_terminal"}
            for rows, paths in ((100, 1048576), (1000, 65536), (1000, 1000000)):
                with self.assertRaises(ValueError):
                    case_result(root, case, {"rows": rows, "paths": paths})

    def test_dataset_runtime_missing_case_stays_missing(self):
        case = next(c for c in workloads() if c["id"] == "heston__mc_terminal")
        with tempfile.TemporaryDirectory() as directory:
            result = case_result(Path(directory), case, {"rows": 1000, "paths": 1048576})
            self.assertEqual(result["status"], "pending")
            self.assertIsNone(result["timings"])
            self.assertFalse(result["environment_eligible_for_tuning"])
            (Path(directory) / case["id"]).mkdir()
            (Path(directory) / "journal.ndjson").write_text(json.dumps({
                "case": case["id"], "status": "unavailable", "reason": "binary not built"}) + "\n")
            result = case_result(Path(directory), case, {"rows": 1000, "paths": 1048576})
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reason"], "binary not built")
            self.assertIsNone(result["timings"])

    def test_dataset_runtime_exports_only_verified_completed_evidence(self):
        for name in ("heston__mc_terminal", "cir__closed_form"):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                case = next(c for c in workloads() if c["id"] == name)
                closed = case["family"] == "closed_form"
                folder = root / name
                folder.mkdir()
                model = root / "source.json"
                model.write_text(json.dumps({"database_id": "source", "row_count": 1000,
                    "models": [{"id": str(i)} for i in range(1000)]}))
                fixture = prepare_inputs({"model": model}, folder / "inputs", [1000],
                                         profile=CATALOGUE_INPUT_PROFILE)
                (folder / "provenance.json").write_text(json.dumps({
                    "fixture": fixture, "binary_sha256": "test-binary", "inputs": {}}))
                (folder / "outcome.json").write_text(json.dumps({
                    "status": "measured", "timing_eligible": False,
                    "timing_ineligibility_reasons": ["test power excursion"]}))
                resources = [{"kernel": "test", "variant": "call",
                    "launch": {"threads_per_block": 128, "grid_block_count": blocks,
                               "dynamic_shared_bytes_per_block": 256},
                    "resources": {"registers_per_thread": 40,
                                  "local_bytes_per_thread": 0,
                                  "static_shared_bytes_per_block": 0},
                    "occupancy": {"theoretical": 1.0}} for blocks in (100, 16, 256)]
                (folder / "resources.json").write_text(json.dumps(resources))
                job = {"id": "native", "rows": 1000, "paths": 1048576}
                row = {"event": "scaling_result", "id": "native", "configuration": {
                    "rows": 1000, "offset": 0, "paths_per_price": 0 if closed else 1048576,
                    "repetitions": 3}, "prices": [1.0] * 1000,
                    "standard_errors": [] if closed else [.1] * 1000,
                    "deterministic_replay": True, "preparation_once_ms": 10,
                    "gpu": {"median_ms": 120}, "raw_host_clock": {"median_ms": 100},
                    "public_api": {"median_ms": 120}, "output_copy": {"median_ms": 2},
                    "publication": {"wall_ms": 3, "native_writer": True},
                    "gpu_samples_ms": [120] * 3, "raw_host_samples_ms": [100] * 3,
                    "api_samples_ms": [120] * 3, "operations_per_sample": 1,
                    "fixed_time_step": "" if closed else "1 / 504",
                    "environment": {}, "device_memory": {}}
                raw = folder / "stdout.ndjson"
                # Incomplete trailing output must not hide the completed record.
                raw.write_text(json.dumps(row) + "\n{")
                result = case_result(root, case, job)
                self.assertEqual(result["timings"]["generation_phase_sum_ms"], 115)
                self.assertEqual(result["paths_per_price"], 0 if closed else 1048576)
                self.assertEqual(len(result["samples"]["gpu_samples_ms"]), 3)
                self.assertEqual(len(result["resources"]), 1)
                self.assertEqual(result["resources"][0]["grid_blocks_min"], 16)
                self.assertEqual(result["resources"][0]["grid_blocks_max"], 256)
                self.assertFalse(result["environment_eligible_for_tuning"])
                self.assertFalse(result["independent_price_certification"])
                self.assertEqual(result["timing_ineligibility_reasons"], ["test power excursion"])
                for field, invalid in (("deterministic_replay", False), ("prices", [1.0]),
                                       ("raw_host_samples_ms", [100]),
                                       ("standard_errors", [float("nan")])):
                    raw.write_text(json.dumps({**row, field: invalid}) + "\n")
                    with self.assertRaises(ValueError):
                        case_result(root, case, job)
                raw.write_text(json.dumps(row) + "\n")
                Path(fixture["by_count"]["1000"]["model"]).write_text("{}")
                with self.assertRaises(ValueError):
                    case_result(root, case, job)

    def test_dataset_runtime_empty_campaign_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text(json.dumps({
                "input_profile": CATALOGUE_INPUT_PROFILE, "selected_jobs": {}}))
            with self.assertRaises(ValueError):
                export_campaign(root)

    def test_catalogue_cli_preserves_order_and_rejects_invalid_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "pricing-scaling").mkdir()
            (build / "pricing-scaling/manifest.json").write_text(json.dumps({
                "schema": "ai_factory_pricing_scaling_v2", "cases": workloads()}))
            selected = ["rough_heston__mc_terminal", "heston__mc_terminal"]
            command = [sys.executable, str(ROOT / "tools/performance/run_pricing_scaling.py"),
                       "--build-dir", str(build), "--input-profile", "catalogue",
                       "--path-counts", "1048576", "--stage", "scaling", "--plan-only"]
            planned = subprocess.run(command + ["--output", str(build / "valid"),
                "--price-counts", "1000", "--cases", *selected], capture_output=True, text=True)
            self.assertEqual(planned.returncode, 0, planned.stderr)
            plan = json.loads((build / "valid/plan.json").read_text())
            self.assertEqual(list(plan["selected_jobs"]), selected)
            self.assertEqual(plan["input_profile"], CATALOGUE_INPUT_PROFILE)
            for label, rows, names in (("bad_rows", "100", selected),
                                      ("duplicate", "1000", selected * 2)):
                failed = subprocess.run(command + ["--output", str(build / label),
                    "--price-counts", rows, "--cases", *names], capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse((build / label).exists())

    def test_public_api_scope_encloses_the_gpu_interval(self):
        source = (ROOT / "tests/performance/pricing_scaling_support.cuh").read_text()
        self.assertIn(
            "std::max(normalized_gpu_ms, normalized_host_ms)", source
        )

    def test_raw_host_clock_is_retained_without_clamping(self):
        source = (ROOT / "tests/performance/pricing_scaling_support.cuh").read_text()
        self.assertIn("const double normalized_host_ms = elapsed_ms(host_start) / operations;", source)
        self.assertIn("raw_host_samples.push_back(raw_host_ms);", source)
        self.assertIn('{"raw_host_clock", timing_json(summarize(raw_host_samples))}', source)
        self.assertIn('{"raw_host_samples_ms", raw_host_samples}', source)
        self.assertIn('{"raw_host_ms", raw_host_ms}', source)


if __name__ == "__main__":
    unittest.main()
