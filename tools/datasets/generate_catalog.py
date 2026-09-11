"""Plan or run native catalogue generators sequentially with explicit resume.

Recipes and the compiled launch inspector remain authoritative. This controller
freezes inputs/binaries, checks staged artifacts and optionally publishes them;
it neither tunes CUDA nor invokes independent price validation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.datasets.artifact_publication import contained_path, digest, publish_pair, save_json
from tools.datasets.dataset_provenance import attach_generation, input_fingerprints, snapshot_sources


def inventory(root: Path, kinds: set[str], models: set[str], targets: set[str]) -> list[dict]:
    sys.path.insert(0, str(root / "tools/codegen/pricing_bindings"))
    from capability_manifest import AVAILABLE_DATASET_SPECS, RNG_DOMAIN_BY_RECIPE

    jobs = []
    for spec in AVAILABLE_DATASET_SPECS:
        if spec.dataset_kind not in kinds or (models and spec.model not in models):
            continue
        if targets and spec.cmake_target not in targets:
            continue
        source = contained_path(root, spec.recipe_path).read_text()
        # Join adjacent C++ literals, including split output paths. No evaluation.
        strings = ["".join(json.loads(token) for token in re.findall(r'"(?:[^"\\]|\\.)*"', group))
                   for group in re.findall(r'"(?:[^"\\]|\\.)*"(?:\s*"(?:[^"\\]|\\.)*")*', source)]
        inputs = sorted({item for item in strings if item.startswith("datasets/")
                         and item.endswith(".json") and item != spec.dataset_path})
        if spec.dataset_kind in {"prices", "price_delta"}:
            if len(inputs) != (3 if spec.curve else 2) or spec.construction != "aligned":
                raise ValueError(f"Unrecognized aligned input contract: {spec.recipe_path}")
            shape = None
        else:
            match = re.search(r'"samples_\d+",\s*([\d\']+)U,\s*([\d\']+)U', source)
            if not match or inputs:
                raise ValueError(f"Unrecognized autonomous sample recipe: {spec.recipe_path}")
            shape = [int(value.replace("'", "")) for value in match.groups()]
        domain = RNG_DOMAIN_BY_RECIPE.get(spec.recipe_path)
        jobs.append({"target": spec.cmake_target, "kind": spec.dataset_kind,
                     "model": spec.model, "recipe": spec.recipe_path,
                     "dataset": spec.dataset_path, "catalog": spec.catalog_yaml_path,
                     "inputs": inputs, "sample_shape": shape,
                     "rng_stream_seeds": {name: domain.seed(name) for name in domain.streams} if domain else {},
                     "declared_method": {"engine": spec.engine, "profile": spec.numerical_profile,
                                         "variant": spec.variant, "construction": spec.construction},
                     "identity": "/".join(value for value in (spec.model, spec.curve, spec.product) if value)})
        if spec.dataset_kind == "price_delta":
            recipe_metadata = str(Path(spec.recipe_path).with_name("recipe.yaml"))
            metadata = yaml.safe_load(contained_path(root, recipe_metadata).read_text())
            jobs[-1].update(recipe_metadata=recipe_metadata,
                            sensitivity=metadata["sensitivity"], time_grid=metadata["time_grid"])
    if not jobs or (targets and targets != {job["target"] for job in jobs}):
        raise ValueError("Empty selection or unknown/excluded generator target")
    return jobs


def finite_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite artifact value")
    if isinstance(value, dict):
        for child in value.values():
            finite_values(child)
    elif isinstance(value, list):
        for child in value:
            finite_values(child)


def check_samples(path: Path, job: dict, catalog: dict) -> None:
    """Verify the native line-streamed sample format without a multi-million-row DOM."""
    with path.open() as stream:
        prefix = []
        for line in stream:
            if line.strip() == '"samples": [':
                break
            prefix.append(line)
        else:
            raise ValueError("Missing native sample array")
        envelope = json.loads("".join(prefix).rstrip().rstrip(",") + "}")
        finite_values(envelope)
        if envelope["row_count"] != job["rows"] or envelope["database_id"] != Path(job["dataset"]).stem:
            raise ValueError("Sample envelope contradicts its recipe")
        construction = envelope["construction"]
        if [construction["parameter_count"], construction["paths_per_parameter"]] != job["sample_shape"]:
            raise ValueError("Sample shape contradicts its recipe")
        bounds = catalog["construction"]["maturity_sampling"]
        count = 0
        for line in stream:
            if line.strip() == "]":
                break
            row = json.loads(line.rstrip().removesuffix(","))
            count += 1
            if line.rstrip().endswith(",") != (count < job["rows"]):
                raise ValueError(f"Invalid sample separator at row {count}")
            finite_values(row)
            days = row["maturity_days"]
            if (row["id"] != f"{count:06d}" or not row["parameters"] or not row["values"]
                    or not isinstance(row["parameters"], dict) or not isinstance(row["values"], dict)
                    or type(days) is not int
                    or not bounds["minimum_days"] <= days <= bounds["maximum_days"]
                    or not math.isclose(row["T"], days / 252, rel_tol=1e-7)):
                raise ValueError(f"Invalid sample row {count}")
        else:
            raise ValueError("Unfinished sample array")
        if count != job["rows"] or stream.read().strip() != "}":
            raise ValueError("Incomplete sample artifact")


def check_outputs(work: Path, job: dict) -> list[dict]:
    catalog = yaml.safe_load(contained_path(work, job["catalog"]).read_text())
    if catalog["database_id"] != Path(job["dataset"]).stem or catalog["row_count"] != job["rows"]:
        raise ValueError("Catalogue identity/row count contradicts the frozen recipe")
    finite_values(catalog)
    if job["kind"] in {"prices", "price_delta"}:
        validation = catalog["validation"]
        relative = Path(job["dataset"]).relative_to("datasets/model")
        reference = Path("validation/datasets/price").joinpath(*(part for part in relative.parts if part != "prices"))
        expected_validation = {"status": "pending", "verified": False}
        if job["kind"] == "prices":
            expected_validation["dataset"] = reference.as_posix()
        if validation != expected_validation:
            raise ValueError("Generation must not claim independent certification")
        expected_paths = job["launch_plan"]["paths_per_price"]
        if catalog["summary"].get("monte_carlo_paths_per_price", 0) != expected_paths:
            raise ValueError("Published MC path count contradicts the compiled production plan")
        document = json.loads(contained_path(work, job["dataset"]).read_text())
        finite_values(document)
        if job["kind"] == "price_delta":
            for artifact in (catalog, document):
                if artifact.get("summary", {}).get("monte_carlo_paths_per_price") != expected_paths:
                    raise ValueError("Price-delta path count contradicts the frozen plan")
                for key in ("threads_per_block", "blocks_per_price"):
                    if key in job["launch_plan"] and artifact.get("summary", {}).get(key) != job["launch_plan"][key]:
                        raise ValueError("Price-delta geometry contradicts the frozen plan")
                for key, value in job["sensitivity"].items():
                    if artifact.get("sensitivity", {}).get(key) != value:
                        raise ValueError("Price-delta sensitivity contradicts the frozen recipe")
                if artifact.get("time_grid") != job["time_grid"]:
                    raise ValueError("Price-delta time grid contradicts the frozen recipe")
                if expected_paths and artifact.get("summary", {}).get("seed") != job["rng_stream_seeds"]["dynamics"]:
                    raise ValueError("Price-delta CRN seed contradicts the price recipe")
        if (document["row_count"] != job["rows"] or len(document["results"]) != job["rows"]
                or document["database_id"] != catalog["database_id"]):
            raise ValueError("Price artifact identity/row count mismatch")
        for index, row in enumerate(document["results"], 1):
            if row["id"] != f"{index:06d}" or type(row["outputs"]["price"]) not in (int, float):
                raise ValueError(f"Invalid price row {index}")
            if job["kind"] == "price_delta":
                if type(row["outputs"].get("delta")) not in (int, float):
                    raise ValueError(f"Missing delta at row {index}")
                delta_error = row["outputs"].get("delta_standard_error", None if expected_paths else 0)
                if type(delta_error) not in (int, float) or delta_error < 0:
                    raise ValueError(f"Invalid delta error at row {index}")
                bump = row["spot_bump"]
                if not (0 < bump["lower"] < bump["upper"]
                        and bump["represented_width"] == bump["upper"] - bump["lower"]):
                    raise ValueError(f"Invalid spot bump at row {index}")
            error = row["outputs"].get("standard_error", None if expected_paths else 0)
            if type(error) not in (int, float) or error < 0:
                raise ValueError(f"Missing or invalid standard error at row {index}")
    else:
        check_samples(contained_path(work, job["dataset"]), job, catalog)
    return [{"path": path, "sha256": digest(contained_path(work, path)),
             "previous_sha256": job["previous"][path]} for path in (job["dataset"], job["catalog"])]


def copy_frozen(source: Path, destination: Path) -> str:
    expected = digest(source)
    if expected is None:
        raise ValueError(f"Missing generator/input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest(destination) != expected or digest(source) != expected:
        raise ValueError(f"File changed while taking its snapshot: {source}")
    return expected


def require_current_build(root: Path, build: Path, jobs: list[dict]) -> None:
    """Use the native build graph to reject missing or stale generation prerequisites."""
    targets = [job["target"] for job in jobs]
    if any(job["kind"] in {"prices", "price_delta"} for job in jobs):
        targets.append("inspect_pricing_launch_plan")
    for path in [build / target for target in targets] + [root / item for job in jobs for item in job["inputs"]]:
        if not path.is_file():
            raise ValueError(f"Missing build/input prerequisite: {path}")
    # Refuse stale binaries. Build explicitly before starting the campaign.
    dry = subprocess.run(["ninja", "-C", str(build), "-n", *targets],
                         text=True, capture_output=True, check=True, env={**os.environ, "LC_ALL": "C"})
    if "no work to do" not in dry.stdout:
        raise ValueError("Generators need rebuilding; run the CMake aggregate build first")


def describe_job(inputs: Path, binaries: Path, job: dict) -> dict:
    """Resolve shape, parameter identity and the compiled plan without CUDA execution."""
    if job["kind"] in {"prices", "price_delta"}:
        counts = {json.loads(contained_path(inputs, name).read_text())["row_count"] for name in job["inputs"]}
        if len(counts) != 1:
            raise ValueError(f"Unaligned parameter inputs: {job['target']}")
        rows = counts.pop()
        plan = json.loads(subprocess.check_output([
            str(binaries / "inspect_pricing_launch_plan"), job["identity"], str(rows)
        ] + (["--price-delta"] if job["kind"] == "price_delta" else []), cwd=inputs, text=True))
        description = {"rows": rows, "launch_plan": plan}
    else:
        description = {"rows": math.prod(job["sample_shape"])}
    return {**description, "semantic_inputs": input_fingerprints(inputs, job["inputs"])}


def freeze(root: Path, build: Path, run: Path, jobs: list[dict], publish: bool) -> dict:
    require_current_build(root, build, jobs)
    run.mkdir(parents=True, exist_ok=False)
    state = {"version": 2, "root": str(root), "build": str(build), "publish": publish,
             "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
             "worktree": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
             "input_hashes": {}, "jobs": jobs}
    # Retain the build configuration and exact controller independently of the
    # mutable checkout. Binary hashes do not prove source-level reproducibility.
    state["build_hashes"] = {name: copy_frozen(build / name, run / "build" / name)
                              for name in ("CMakeCache.txt", "build.ninja")}
    state["controller_hashes"] = {
        name: copy_frozen(contained_path(root, name), contained_path(run / "sources", name))
        for name in ("tools/datasets/generate_catalog.py", "tools/datasets/artifact_publication.py",
                     "tools/datasets/dataset_provenance.py")}
    state["source_archive_sha256"] = snapshot_sources(root, run / "sources.tar.gz")
    if any(job["kind"] in {"prices", "price_delta"} for job in jobs):
        state["inspector_sha256"] = copy_frozen(build / "inspect_pricing_launch_plan",
                                                run / "bin" / "inspect_pricing_launch_plan")
    for relative in sorted({item for job in jobs for item in job["inputs"]}):
        state["input_hashes"][relative] = copy_frozen(contained_path(root, relative),
                                                      contained_path(run / "inputs", relative))
    for job in jobs:
        job["binary_sha256"] = copy_frozen(build / job["target"], run / "bin" / job["target"])
        job["recipe_sha256"] = copy_frozen(root / job["recipe"], contained_path(run / "sources", job["recipe"]))
        if job.get("recipe_metadata"):
            job["recipe_metadata_sha256"] = copy_frozen(root / job["recipe_metadata"],
                contained_path(run / "sources", job["recipe_metadata"]))
        job["previous"] = {path: digest(contained_path(root, path)) for path in (job["dataset"], job["catalog"])}
        job.update(describe_job(run / "inputs", run / "bin", job))
        job["state"] = "pending"
    require_current_build(root, build, jobs)
    save_json(run / "campaign.json", state)
    return state


def run_generator(binary: Path, work: Path, logs: Path) -> float:
    started = time.perf_counter()
    with (logs / "stdout.log").open("w") as out, (logs / "stderr.log").open("w") as err:
        process = subprocess.Popen([str(binary)], cwd=work, stdout=out, stderr=err, start_new_session=True)
        try:
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode:
        raise RuntimeError(f"Generator exited with code {returncode}; see {logs}")
    return time.perf_counter() - started


def gpu_observation() -> dict:
    """Record environment without turning telemetry into an execution veto."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem",
            "--format=csv"], text=True, capture_output=True, timeout=10)
        return {"unix_time": time.time(), "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unix_time": time.time(), "unavailable": str(error)}


def execute(run: Path, state: dict) -> None:
    root = Path(state["root"])
    if digest(run / "sources.tar.gz") != state["source_archive_sha256"]:
        raise ValueError("Frozen source archive changed")
    for name, expected in state["build_hashes"].items():
        if digest(contained_path(run / "build", name)) != expected:
            raise ValueError(f"Frozen build configuration changed: {name}")
    for index, job in enumerate(state["jobs"], 1):
        directory = contained_path(run / "jobs", job["target"])
        binary = contained_path(run / "bin", job["target"])
        if digest(binary) != job["binary_sha256"]:
            raise ValueError(f"Frozen executable changed: {binary}")
        if digest(contained_path(run / "sources", job["recipe"])) != job["recipe_sha256"]:
            raise ValueError("Frozen recipe changed")
        if job.get("recipe_metadata") and digest(contained_path(run / "sources", job["recipe_metadata"])) != job["recipe_metadata_sha256"]:
            raise ValueError("Frozen recipe metadata changed")
        for item in job["inputs"]:
            if digest(contained_path(run / "inputs", item)) != state["input_hashes"][item]:
                raise ValueError(f"Frozen input changed: {item}")
        if job["state"] == "complete":
            owner = root if state["publish"] else Path(job["work"])
            if any(digest(contained_path(owner, item["path"])) != item["sha256"] for item in job["artifacts"]):
                raise ValueError(f"Completed output changed: {job['target']}")
            continue
        print(f"[{index}/{len(state['jobs'])}] {job['target']} ({job['state']})", flush=True)
        try:
            if job["state"] != "staged":
                # An explicit resume starts a fresh attempt for an unfinished dataset.
                attempt = job.get("attempt", 0) + 1
                logs = directory / f"attempt-{attempt:03d}"
                work = logs / "work"
                work.mkdir(parents=True, exist_ok=False)
                job.update(attempt=attempt, work=str(work), state="running")
                save_json(run / "campaign.json", state)
                old_bytes = sum((root / path).stat().st_size for path, value in job["previous"].items() if value)
                # Conservative planning estimate, not a promise of final JSON size.
                required = job["rows"] * 2048 + old_bytes + 1024**3
                if shutil.disk_usage(work).free < required:
                    raise RuntimeError(f"Insufficient disk margin: need about {required} bytes for this job")
                for item in job["inputs"]:
                    copy_frozen(contained_path(run / "inputs", item), contained_path(work, item))
                job["gpu_before"] = gpu_observation()
                save_json(run / "campaign.json", state)
                try:
                    job["generation_wall_seconds"] = run_generator(binary, work, logs)
                finally:
                    job["gpu_after"] = gpu_observation()
                for item in job["inputs"]:
                    if digest(contained_path(work, item)) != state["input_hashes"][item]:
                        raise ValueError(f"Generator modified a parameter input: {item}")
                started = time.perf_counter()
                job["artifacts"] = check_outputs(work, job)
                attach_generation(work, job, state)
                # The publication journal must cover the enriched YAML, not
                # the native intermediate. JSON bytes remain untouched.
                job["artifacts"][1]["sha256"] = digest(contained_path(work, job["catalog"]))
                job["artifact_check_seconds"] = time.perf_counter() - started
                job["state"] = "staged"
                save_json(run / "campaign.json", state)
            if state["publish"]:
                started = time.perf_counter()
                try:
                    publish_pair(root, Path(job["work"]), directory / "publication.json", job["artifacts"])
                finally:
                    job["publication_seconds"] = job.get("publication_seconds", 0) + time.perf_counter() - started
            job["state"] = "complete"
            job.pop("last_error", None)
            save_json(run / "campaign.json", state)
        except BaseException as error:
            job["last_error"] = str(error) or type(error).__name__
            # A staged job retains its publication journal; do not regenerate it.
            if job["state"] != "staged":
                job["state"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            save_json(run / "campaign.json", state)
            raise


def interrupt_campaign(*_arguments) -> None:
    raise KeyboardInterrupt("campaign interrupted; resume explicitly")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build-dev")
    parser.add_argument("--kind", choices=("prices", "price_delta", "samples", "all"), default="prices")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="otherwise only inspect the selection")
    parser.add_argument("--publish", action="store_true", help="replace verified staged outputs, retaining backups")
    parser.add_argument("--resume", action="store_true", help="explicitly resume the frozen campaign")
    arguments = parser.parse_args()
    if arguments.resume and (not arguments.execute or not arguments.run_dir):
        parser.error("--resume requires --execute and --run-dir")
    if arguments.publish and not arguments.execute:
        parser.error("--publish requires --execute")
    if arguments.resume:
        state = json.loads((arguments.run_dir / "campaign.json").read_text())
        if state["version"] != 2 or state["root"] != str(ROOT):
            raise ValueError("Unsupported campaign version or repository; retain its original controller")
        for name, expected in state["controller_hashes"].items():
            if digest(contained_path(ROOT, name)) != expected:
                raise ValueError(f"Controller changed since this campaign was frozen: {name}")
        if arguments.model or arguments.target or arguments.kind != "prices" or (arguments.publish and not state["publish"]):
            raise ValueError("Resume uses the frozen selection and publication policy; do not override them")
        jobs = state["jobs"]
    else:
        kinds = {"prices", "price_delta", "samples"} if arguments.kind == "all" else {arguments.kind}
        jobs = inventory(ROOT, kinds, set(arguments.model), set(arguments.target))
    if not arguments.execute:
        print(json.dumps({"job_count": len(jobs), "jobs": jobs}, indent=2))
        return 0
    if arguments.run_dir is None:
        parser.error("--execute requires a new --run-dir (or --resume)")
    build = Path(state["build"]) if arguments.resume else arguments.build.resolve()
    # One lock for the repository, including controllers using different builds.
    lock_path = ROOT / "datasets" / ".generation-campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = arguments.run_dir.resolve()
        if not arguments.resume:
            state = freeze(ROOT, build, run, jobs, arguments.publish)
        signal.signal(signal.SIGTERM, interrupt_campaign)
        execute(run, state)
    print("Campaign complete. Independent price validation was not run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(f"Generation stopped: {error or type(error).__name__}", file=sys.stderr)
        raise SystemExit(1)
