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
import struct
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.datasets.artifact_publication import contained_path, digest, publish_pair, save_json
from tools.datasets.dataset_provenance import (
    attach_generation,
    fingerprint,
    input_fingerprints,
    snapshot_sources,
)
from tools.datasets.metadata_schemas import validate_document


def save_campaign(path: Path, state: dict) -> None:
    """Persist only campaign states accepted by the shared machine contract."""
    validate_document(state, "campaign", path)
    save_json(path, state)


def source_family(spec) -> str | None:
    """Return the equity source family declared by a model dataset spec."""
    parts = Path(spec.source_prefix).parts
    if len(parts) >= 4 and parts[:2] == ("model", "equity"):
        return parts[2]
    return None


def select_specs(
    root: Path,
    specs,
    kinds: set[str],
    models: set[str],
    targets: set[str],
    *,
    asset_classes: set[str] | None = None,
    model_families: set[str] | None = None,
    constructions: set[str] | None = None,
    skip_published: bool = False,
    catalog_only: bool = False,
) -> list:
    """Filter manifest specs and optionally exclude complete published pairs."""
    asset_classes = asset_classes or set()
    model_families = model_families or set()
    constructions = constructions or set()
    selected = []
    matched_targets = set()
    skipped_published = 0
    for spec in specs:
        if spec.dataset_kind not in kinds:
            continue
        if models and spec.model not in models:
            continue
        if targets and spec.cmake_target not in targets:
            continue
        if asset_classes and spec.asset_class not in asset_classes:
            continue
        if model_families and source_family(spec) not in model_families:
            continue
        if constructions and spec.construction not in constructions:
            continue
        if catalog_only and not contained_path(root, spec.generator_path).is_relative_to(
            root.resolve() / "catalog"
        ):
            continue
        if (root / "catalog/manifest.json").is_file():
            generator = contained_path(root, spec.generator_path)
            if not generator.is_file() and "work/catalog/" in generator.as_posix():
                # The local-only branch is intentionally absent from GitHub.
                continue
        matched_targets.add(spec.cmake_target)
        if skip_published:
            dataset_exists = contained_path(root, spec.dataset_path).exists()
            generation_exists = contained_path(root, spec.generation_yaml_path).exists()
            if dataset_exists != generation_exists:
                raise ValueError(
                    f"Partial published pair for {spec.cmake_target}: "
                    f"{spec.dataset_path}, {spec.generation_yaml_path}"
                )
            if dataset_exists:
                skipped_published += 1
                continue
        selected.append(spec)
    unknown_targets = targets - matched_targets
    if unknown_targets:
        raise ValueError(
            "Targets outside the selection: " + ", ".join(sorted(unknown_targets))
        )
    if not selected and not (skip_published and skipped_published):
        raise ValueError("Empty dataset generator selection")
    return selected


def inventory(
    root: Path,
    kinds: set[str],
    models: set[str],
    targets: set[str],
    *,
    asset_classes: set[str] | None = None,
    model_families: set[str] | None = None,
    constructions: set[str] | None = None,
    skip_published: bool = False,
    catalog_only: bool = False,
) -> list[dict]:
    sys.path.insert(0, str(root / "tools/codegen/pricing_bindings"))
    from capability_manifest import (
        AVAILABLE_DATASET_SPECS,
        RNG_DOMAIN_BY_GENERATOR,
        PRICE_GRADIENT_SOURCE_BY_GENERATOR,
        rng_mapping_version,
    )

    jobs = []
    selected = select_specs(
        root,
        AVAILABLE_DATASET_SPECS,
        kinds,
        models,
        targets,
        asset_classes=asset_classes,
        model_families=model_families,
        constructions=constructions,
        skip_published=skip_published,
        catalog_only=catalog_only,
    )
    for spec in selected:
        source = contained_path(root, spec.generator_path).read_text()
        recipe_path = spec.recipe_yaml_path
        recipe = yaml.safe_load(contained_path(root, recipe_path).read_text())
        if not isinstance(recipe, dict):
            raise ValueError(f"Invalid recipe document: {recipe_path}")
        validate_document(recipe, "recipe", recipe_path)
        if recipe.get("dataset_id") != spec.dataset_id:
            raise ValueError(f"Recipe identity contradicts manifest: {recipe_path}")
        if recipe.get("output") != {"path": spec.dataset_path, "format": "json"}:
            raise ValueError(f"Recipe output contradicts manifest: {recipe_path}")
        if recipe.get("generation_output") != spec.generation_yaml_path:
            raise ValueError(f"Generation output contradicts manifest: {recipe_path}")
        if recipe.get("url") != spec.url:
            raise ValueError(f"Recipe URL contradicts manifest: {recipe_path}")
        if recipe.get("row_count") != spec.row_count:
            raise ValueError(f"Recipe row count contradicts manifest: {recipe_path}")
        if spec.dataset_kind == "price_gradients":
            # Price-only recipes can live in a local work catalogue. Validate
            # the gradient against the typed source capability, which remains
            # available in a clean delivery checkout.
            price_source = PRICE_GRADIENT_SOURCE_BY_GENERATOR[spec.generator_path]
            source_path = recipe["sensitivity"]["source_price_recipe"]
            if source_path != price_source.recipe_yaml_path:
                raise ValueError(
                    f"Sensitivity source contradicts the capability manifest: {recipe_path}"
                )
            source_domain = RNG_DOMAIN_BY_GENERATOR.get(price_source.generator_path)
            expected_rng = "philox" if source_domain else None
            if recipe.get("random_number_generator") != expected_rng:
                raise ValueError(
                    f"Sensitivity and source random generators differ: {recipe_path}"
                )
            if source_domain and recipe.get("seeds", {}).get("dynamics") != source_domain.seed("dynamics"):
                raise ValueError(
                    f"Sensitivity seed contradicts the source random domain: {recipe_path}"
                )
        # Join adjacent C++ literals, including split output paths. No evaluation.
        strings = ["".join(json.loads(token) for token in re.findall(r'"(?:[^"\\]|\\.)*"', group))
                   for group in re.findall(r'"(?:[^"\\]|\\.)*"(?:\s*"(?:[^"\\]|\\.)*")*', source)]
        inputs = sorted({item for item in strings if item.startswith("datasets/")
                         and item.endswith(".json") and item != spec.dataset_path})
        if spec.dataset_kind in {"prices", "price_gradients"}:
            if (len(inputs) != (3 if spec.curve else 2)
                    or spec.construction not in {"aligned", "cartesian"}):
                raise ValueError(f"Unrecognized price input contract: {spec.generator_path}")
            shape = None
        else:
            shape_metadata = recipe.get("shape", {})
            shape = [
                shape_metadata.get("parameter_count"),
                shape_metadata.get("paths_per_parameter"),
            ]
            if any(type(value) is not int or value <= 0 for value in shape) or inputs:
                raise ValueError(f"Unrecognized autonomous sample recipe: {spec.generator_path}")
        domain = RNG_DOMAIN_BY_GENERATOR.get(spec.generator_path)
        jobs.append({"target": spec.cmake_target, "kind": spec.dataset_kind,
                     "model": spec.model, "generator": spec.generator_path,
                     "recipe": recipe_path, "generation": spec.generation_yaml_path,
                     "dataset": spec.dataset_path,
                     "url": spec.url,
                     "inputs": inputs, "sample_shape": shape,
                     "rng_stream_seeds": {name: domain.seed(name) for name in domain.streams} if domain else {},
                     "row_count": recipe["row_count"],
                     "declared_method": {
                         "engine": spec.engine,
                         "profile": spec.numerical_profile,
                         "variant": spec.variant,
                         "construction": spec.construction,
                         "has_curve": spec.curve is not None,
                         "random_number_generator": (
                             {
                                 "name": "philox",
                                 "mapping_version": rng_mapping_version(spec.model),
                             }
                             if domain else None
                         ),
                     },
                     "identity": "/".join(value for value in (spec.model, spec.curve, spec.product) if value)})
        if spec.dataset_kind in {"prices", "price_gradients"}:
            jobs[-1]["paths_per_price"] = recipe.get("paths_per_price")
        if spec.dataset_kind in {"price_gradients"}:
            time_key = "time_representation" if "time_representation" in recipe else "time_grid"
            jobs[-1].update(sensitivity=recipe["sensitivity"], time_key=time_key,
                            time_configuration=recipe[time_key],
                            exercise_replay=recipe.get("exercise_replay"),
                            preparation=recipe.get("preparation", {}))
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


def check_samples(path: Path, job: dict, recipe: dict) -> None:
    """Verify the native line-streamed sample format without a multi-million-row DOM."""
    with path.open() as stream:
        prefix = []
        for line in stream:
            if line.strip() == '"results": [':
                break
            prefix.append(line)
        else:
            raise ValueError("Missing native results array")
        envelope = json.loads("".join(prefix).rstrip().rstrip(",") + "}")
        finite_values(envelope)
        expected_keys = {
            "database_id", "model_family", "catalog", "url", "row_count",
            "time_convention", "timing",
        }
        if set(envelope) != expected_keys:
            raise ValueError("Sample envelope contains non-public generation metadata")
        if envelope["row_count"] != job["rows"] or envelope["database_id"] != Path(job["dataset"]).stem:
            raise ValueError("Sample envelope contradicts its recipe")
        bounds = recipe["maturity_sampling"]
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
    recipe = yaml.safe_load(contained_path(work, job["recipe"]).read_text())
    generation = yaml.safe_load(contained_path(work, job["generation"]).read_text())
    if not isinstance(recipe, dict) or not isinstance(generation, dict):
        raise ValueError("Recipe and generation receipt must be mappings")
    validate_document(recipe, "recipe", job["recipe"])
    if (recipe.get("dataset_id") != Path(job["dataset"]).stem
            or recipe.get("output") != {"path": job["dataset"], "format": "json"}
            or recipe.get("url") != job["url"]):
        raise ValueError("Recipe identity/output/URL contradicts the frozen selection")
    if (generation.get("schema_version") != 1
            or generation.get("status") != "complete"
            or generation.get("artifact", {}).get("row_count") != job["rows"]
            or not isinstance(generation.get("execution"), dict)
            or not isinstance(generation.get("timing"), dict)):
        raise ValueError("Invalid native generation receipt")
    forbidden = {
        "database_id", "dataset_id", "model_dataset", "curve_dataset",
        "product_dataset", "price_construction", "validation", "outputs",
        "seeds", "sensitivity", "time_grid", "time_representation",
    }
    if forbidden & generation.keys():
        raise ValueError("Generation receipt repeats recipe or validation fields")
    finite_values(generation)
    if job["kind"] in {"prices", "price_gradients"}:
        expected_paths = job["launch_plan"]["paths_per_price"]
        if generation["execution"].get("paths_per_price", 0) != expected_paths:
            raise ValueError("Published MC path count contradicts the compiled production plan")
        native_plan = generation["execution"].get("launch_plan")
        if (isinstance(native_plan, dict)
                and native_plan.get("profile_id") != job["launch_plan"].get("profile_id")):
            raise ValueError("Native launch profile contradicts the compiled production plan")
        expected_replay = job.get("exercise_replay")
        if (job["kind"] == "price_gradients"
                and generation["execution"].get("exercise_replay")
                != expected_replay):
            raise ValueError(
                "Generation receipt exercise replay contradicts frozen recipe"
            )
        document = json.loads(contained_path(work, job["dataset"]).read_text())
        finite_values(document)
        if document.get("url") != job["url"]:
            raise ValueError("Dataset URL contradicts the frozen recipe")
        expected_construction = (
            {
                "method": "Cartesian product",
                "order": (
                    "model, curve, product"
                    if job["declared_method"]["has_curve"]
                    else "model, product"
                ),
            }
            if job["declared_method"]["construction"] == "cartesian"
            else {"method": "Aligned"}
        )
        if document.get("price_construction", expected_construction) != expected_construction:
            raise ValueError("Price construction contradicts the frozen recipe")
        if job["kind"] == "price_gradients":
            from tools.datasets.price_gradients.contract import check_outputs as check_gradients
            check_gradients(job, document, document)
        if (document["row_count"] != job["rows"] or len(document["results"]) != job["rows"]
                or document["database_id"] != recipe["dataset_id"]):
            raise ValueError("Price artifact identity/row count mismatch")
        for index, row in enumerate(document["results"], 1):
            if row["id"] != f"{index:06d}" or type(row["outputs"]["price"]) not in (int, float):
                raise ValueError(f"Invalid price row {index}")
            error = row["outputs"].get("standard_error", None if expected_paths else 0)
            if type(error) not in (int, float) or error < 0:
                raise ValueError(f"Missing or invalid standard error at row {index}")
    else:
        document_url = None
        with contained_path(work, job["dataset"]).open() as stream:
            for line in stream:
                if '"url"' in line:
                    document_url = json.loads("{" + line.strip().rstrip(",") + "}")["url"]
                    break
        if document_url != job["url"]:
            raise ValueError("Dataset URL contradicts the frozen recipe")
        check_samples(contained_path(work, job["dataset"]), job, recipe)
    return [{"path": path, "sha256": digest(contained_path(work, path)),
             "previous_sha256": job["previous"][path]}
            for path in (job["dataset"], job["generation"])]


def copy_frozen(source: Path, destination: Path) -> str:
    expected = digest(source)
    if expected is None:
        raise ValueError(f"Missing generator/input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if digest(destination) != expected or digest(source) != expected:
        raise ValueError(f"File changed while taking its snapshot: {source}")
    return expected


def build_graph_files(build: Path) -> tuple[str, ...]:
    """Return the CMake files that identify the native build graph."""
    cache = (build / "CMakeCache.txt").read_text()
    match = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.+)$", cache, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing CMake generator in {build / 'CMakeCache.txt'}")
    generator = match.group(1)
    if generator == "Ninja":
        return ("CMakeCache.txt", "build.ninja")
    if generator == "Unix Makefiles":
        return ("CMakeCache.txt", "Makefile", "CMakeFiles/Makefile2")
    raise ValueError(f"Unsupported CMake generator for campaign freshness check: {generator}")


def cmake_make_program(build: Path) -> str:
    cache = (build / "CMakeCache.txt").read_text()
    match = re.search(r"^CMAKE_MAKE_PROGRAM:[^=]+=(.+)$", cache, re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing CMake make program in {build / 'CMakeCache.txt'}")
    return match.group(1)


def require_current_build(root: Path, build: Path, jobs: list[dict],
                          input_overrides: dict[str, Path] | None = None) -> None:
    """Use the native build graph to reject missing or stale generation prerequisites."""
    input_overrides = input_overrides or {}
    targets = [job["target"] for job in jobs]
    if any(job["kind"] in {"prices", "price_gradients"} for job in jobs):
        targets.append("inspect_pricing_launch_plan")
    inputs = [input_overrides.get(item, contained_path(root, item))
              for job in jobs for item in job["inputs"]]
    for path in [build / target for target in targets] + inputs:
        if not path.is_file():
            raise ValueError(f"Missing build/input prerequisite: {path}")
    graph = build_graph_files(build)
    if graph[1] == "build.ninja":
        dry = subprocess.run(["ninja", "-C", str(build), "-n", *targets],
                             text=True, capture_output=True, check=True,
                             env={**os.environ, "LC_ALL": "C"})
        stale = "no work to do" not in dry.stdout
    else:
        # CMake's outer Makefile has phony targets: make -q reports stale even
        # when every executable is current. Dry-run the concrete dependency
        # graph and look for an actual compile, generated file or link step.
        dry = subprocess.run(
            [cmake_make_program(build), "-C", str(build), "-n", "-f", "CMakeFiles/Makefile2",
             *(f"CMakeFiles/{target}.dir/all" for target in targets)],
            text=True, capture_output=True, check=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        stale = any(re.search(r'"(?:Building|Linking|Generating) ', line)
                    for line in dry.stdout.splitlines())
    if stale:
        raise ValueError("Generators need rebuilding; run the CMake aggregate build first")


def compile_selected(root: Path, build: Path, jobs: list[dict], parallel_jobs: int) -> None:
    """Compile only the selected generators and their shared launch inspector."""
    if parallel_jobs <= 0:
        raise ValueError("Compile jobs must be positive")
    targets = {job["target"] for job in jobs}
    if any(job["kind"] in {"prices", "price_gradients"} for job in jobs):
        targets.add("inspect_pricing_launch_plan")
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            *sorted(targets),
            f"-j{parallel_jobs}",
        ],
        cwd=root,
        env={**os.environ, "CCACHE_DISABLE": "1"},
        check=True,
    )


def describe_job(inputs: Path, binaries: Path, job: dict) -> dict:
    """Resolve shape, parameter identity and the compiled plan without CUDA execution."""
    if job["kind"] in {"prices", "price_gradients"}:
        input_counts = [json.loads(contained_path(inputs, name).read_text())["row_count"]
                        for name in job["inputs"]]
        counts = set(input_counts)
        construction = job["declared_method"].get("construction", "aligned")
        if construction == "aligned" and len(counts) != 1:
            raise ValueError(f"Unaligned parameter inputs: {job['target']}")
        rows = (math.prod(input_counts)
                if construction == "cartesian"
                else input_counts[0])
        inspector = [str(binaries / "inspect_pricing_launch_plan"), job["identity"], str(rows)]
        if job.get("paths_per_price"):
            inspector.append(str(job["paths_per_price"]))
        if job.get("maximum_resident_prices"):
            if not job.get("paths_per_price"):
                raise ValueError(
                    "maximum_resident_prices requires paths_per_price: "
                    f"{job['target']}"
                )
            inspector.append(str(job["maximum_resident_prices"]))
        if job["kind"] == "price_gradients":
            inspector += [
                "--price-gradients",
                str(len(job["sensitivity"]["parameters"])),
            ]
        if job["kind"] == "price_gradients":
            orders = job["sensitivity"].get("orders", [])
            if "mixed_second" in orders:
                inspector.append("--mixed")
            elif "diagonal_second" in orders:
                inspector.append("--diagonal")
                strategy = job.get("sensitivity_strategy", "mono")
                if strategy not in {"mono", "node_graph"}:
                    raise ValueError(
                        f"Unknown diagonal sensitivity strategy: {strategy}"
                    )
                if strategy == "node_graph":
                    inspector.append("--node-graph")
        plan = json.loads(subprocess.check_output(inspector, cwd=inputs, text=True))
        description = {"rows": rows, "launch_plan": plan}
    else:
        description = {"rows": math.prod(job["sample_shape"])}
    if description["rows"] != job["row_count"]:
        raise ValueError(f"Input cardinality contradicts recipe: {job['target']}")
    return {**description, "semantic_inputs": input_fingerprints(inputs, job["inputs"])}


def freeze(root: Path, build: Path, run: Path, jobs: list[dict], publish: bool,
           input_overrides: dict[str, Path] | None = None) -> dict:
    input_overrides = input_overrides or {}
    declared_inputs = {item for job in jobs for item in job["inputs"]}
    if set(input_overrides) - declared_inputs:
        raise ValueError("Input override is not used by the selected generators")
    require_current_build(root, build, jobs, input_overrides)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    worktree = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    )
    if publish and worktree:
        raise ValueError(
            "Publishing requires a clean Git worktree; commit the exact sources first"
        )
    run.mkdir(parents=True, exist_ok=False)
    state = {"version": 4, "root": str(root), "build": str(build), "publish": publish,
             "revision": revision, "worktree": worktree,
             "input_hashes": {}, "jobs": jobs}
    # Retain the build configuration and exact controller independently of the
    # mutable checkout. Binary hashes do not prove source-level reproducibility.
    state["build_hashes"] = {name: copy_frozen(build / name, run / "build" / name)
                              for name in build_graph_files(build)}
    state["controller_hashes"] = {
        name: copy_frozen(contained_path(root, name), contained_path(run / "sources", name))
        for name in ("tools/datasets/generate_catalog.py", "tools/datasets/artifact_publication.py",
                     "tools/datasets/dataset_provenance.py",
                     "tools/datasets/metadata_schemas.py",
                     "tools/datasets/schemas/campaign.schema.yaml",
                     "tools/datasets/schemas/generation.schema.yaml",
                     "tools/datasets/schemas/recipe.schema.yaml",
                     "tools/datasets/schemas/validation.schema.yaml")}
    state["source_archive_sha256"] = snapshot_sources(root, run / "sources.tar.gz")
    if any(job["kind"] in {"prices", "price_gradients"} for job in jobs):
        state["inspector_sha256"] = copy_frozen(build / "inspect_pricing_launch_plan",
                                                run / "bin" / "inspect_pricing_launch_plan")
        launch_manifest = "cmake/generated/PricingCapabilityManifest.json"
        state["launch_manifest_sha256"] = copy_frozen(
            contained_path(root, launch_manifest),
            contained_path(run / "inputs", launch_manifest),
        )
    state["input_origins"] = {}
    for relative in sorted(declared_inputs):
        source = input_overrides.get(relative, contained_path(root, relative))
        state["input_hashes"][relative] = copy_frozen(source,
                                                      contained_path(run / "inputs", relative))
        if relative in input_overrides:
            state["input_origins"][relative] = str(source.resolve())
    for job in jobs:
        job["binary_sha256"] = copy_frozen(build / job["target"], run / "bin" / job["target"])
        job["generator_sha256"] = copy_frozen(
            contained_path(root, job["generator"]), contained_path(run / "sources", job["generator"])
        )
        job["recipe_sha256"] = copy_frozen(
            contained_path(root, job["recipe"]), contained_path(run / "sources", job["recipe"])
        )
        job["previous"] = {path: digest(contained_path(root, path))
                           for path in (job["dataset"], job["generation"])}
        job.update(describe_job(run / "inputs", run / "bin", job))
        if job["kind"] in {"prices", "price_gradients"} and job["launch_plan"]["paths_per_price"]:
            checkpoint_contract = {
                "schema_version": 1,
                "kind": job["kind"],
                "target": job["target"],
                "rows": job["rows"],
                "binary_sha256": job["binary_sha256"],
                "generator_sha256": job["generator_sha256"],
                "recipe_sha256": job["recipe_sha256"],
                "input_hashes": {name: state["input_hashes"][name] for name in job["inputs"]},
                "launch_plan": job["launch_plan"],
                "rng_stream_seeds": job["rng_stream_seeds"],
                "sensitivity": job.get("sensitivity"),
                "exercise_replay": job.get("exercise_replay"),
                "time_grid": job.get("time_grid"),
                "preparation": job.get("preparation"),
            }
            job["checkpoint"] = f"jobs/{job['target']}/checkpoint"
            job["checkpoint_id"] = fingerprint(checkpoint_contract)
        job["state"] = "pending"
    require_current_build(root, build, jobs, input_overrides)
    save_campaign(run / "campaign.json", state)
    return state


def compare_matching_prices(state: dict, job: dict, work: Path) -> dict | None:
    """Record central-price agreement without vetoing valid native generation.

    Closed-form scalar and cooperative kernels sum FP32 bond legs in different
    orders. Their comparison is a qualification diagnostic, not an identity
    requirement for two independently generated datasets.
    """
    if job["kind"] != "price_gradients":
        return None
    source_recipe = job["sensitivity"]["source_price_recipe"]
    peers = [candidate for candidate in state["jobs"]
             if candidate["kind"] == "prices" and candidate["recipe"] == source_recipe]
    if not peers:
        return None
    if len(peers) != 1:
        raise ValueError(f"Ambiguous source price recipe: {source_recipe}")
    peer = peers[0]
    if peer["state"] != "complete" or peer["inputs"] != job["inputs"]:
        raise ValueError(f"Source price is unfinished or uses different inputs: {peer['target']}")
    source_dataset = contained_path(Path(peer["work"]), peer["dataset"])
    source_digest = digest(source_dataset)
    if "artifacts" in peer and source_digest != peer["artifacts"][0]["sha256"]:
        raise ValueError(f"Source price dataset changed after staging: {peer['target']}")
    price_rows = json.loads(source_dataset.read_text())["results"]
    gradient_rows = json.loads(contained_path(work, job["dataset"]).read_text())["results"]
    if len(price_rows) != len(gradient_rows):
        raise ValueError(f"Central price row count differs: {job['target']}")
    different = outside_budget = 0
    maximum_absolute = maximum_budget_ratio = 0.0
    for index, (price, gradient) in enumerate(zip(price_rows, gradient_rows), 1):
        if price["id"] != gradient["id"]:
            raise ValueError(f"Central price row identity differs at row {index}")
        left = price["outputs"]["price"]
        right = gradient["outputs"]["price"]
        difference = abs(left - right)
        budget = 2e-6 + 2e-5 * max(abs(left), abs(right))
        different += struct.pack("!d", left) != struct.pack("!d", right)
        outside_budget += difference > budget
        maximum_absolute = max(maximum_absolute, difference)
        maximum_budget_ratio = max(maximum_budget_ratio, difference / budget)
    return {
        "source_target": peer["target"],
        "source_dataset_sha256": source_digest,
        "matched_rows": len(price_rows),
        "comparison": "fp32_cross_kernel_budget_v1",
        "different_rows": different,
        "outside_budget_rows": outside_budget,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_budget_ratio": maximum_budget_ratio,
        "budget": "2e-6 + 2e-5 * max(abs(price_only), abs(gradient_central))",
        "status": "within_budget" if outside_budget == 0 else "review_required",
    }


def append_journal(path: Path, event: str, **details: object) -> None:
    """Append a timestamped campaign event to one dataset attempt."""
    with path.open("a") as output:
        output.write(json.dumps({"event": event, "unix_time": time.time(), **details}) + "\n")


def run_generator(
    binary: Path,
    work: Path,
    logs: Path,
    progress: Path,
    checkpoint: Path | None = None,
    checkpoint_id: str | None = None,
) -> float:
    started = time.perf_counter()
    journal = logs / "progress.jsonl"
    if (checkpoint is None) != (checkpoint_id is None):
        raise ValueError("A checkpoint directory and identity must be provided together")
    environment = {
        **os.environ,
        "AI_FACTORY_GENERATION_PROGRESS": str(progress.resolve()),
        "AI_FACTORY_GENERATION_PROGRESS_LOG": str(journal.resolve()),
    }
    if checkpoint is not None:
        environment.update(
            AI_FACTORY_GENERATION_CHECKPOINT_DIR=str(checkpoint.resolve()),
            AI_FACTORY_GENERATION_CHECKPOINT_ID=checkpoint_id,
        )
    append_journal(
        journal,
        "generator_started",
        checkpoint=str(checkpoint) if checkpoint is not None else None,
    )
    with (logs / "stdout.log").open("w") as out, (logs / "stderr.log").open("w") as err:
        process = subprocess.Popen(
            [str(binary)],
            cwd=work,
            stdout=out,
            stderr=err,
            start_new_session=True,
            env=environment,
        )
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
            append_journal(journal, "generator_interrupted", returncode=process.returncode)
            raise
    append_journal(journal, "generator_exited", returncode=returncode,
                   wall_seconds=time.perf_counter() - started)
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
    verify_verifier_amendment(run, state)
    for name, expected in state["build_hashes"].items():
        if digest(contained_path(run / "build", name)) != expected:
            raise ValueError(f"Frozen build configuration changed: {name}")
    for index, job in enumerate(state["jobs"], 1):
        directory = contained_path(run / "jobs", job["target"])
        binary = contained_path(run / "bin", job["target"])
        if digest(binary) != job["binary_sha256"]:
            raise ValueError(f"Frozen executable changed: {binary}")
        if digest(contained_path(run / "sources", job["generator"])) != job["generator_sha256"]:
            raise ValueError("Frozen generator changed")
        if digest(contained_path(run / "sources", job["recipe"])) != job["recipe_sha256"]:
            raise ValueError("Frozen recipe changed")
        for item in job["inputs"]:
            if digest(contained_path(run / "inputs", item)) != state["input_hashes"][item]:
                raise ValueError(f"Frozen input changed: {item}")
        if job["state"] == "complete":
            owner = root if state["publish"] else Path(job["work"])
            if any(digest(contained_path(owner, item["path"])) != item["sha256"] for item in job["artifacts"]):
                raise ValueError(f"Completed output changed: {job['target']}")
            continue
        print(f"[{index}/{len(state['jobs'])}] {job['target']} ({job['state']})", flush=True)
        attempt_journal = None
        checkpoint = contained_path(run, job["checkpoint"]) if job.get("checkpoint") else None
        try:
            if job["state"] != "staged":
                # An explicit resume starts a fresh attempt for an unfinished dataset.
                attempt = job.get("attempt", 0) + 1
                logs = directory / f"attempt-{attempt:03d}"
                attempt_journal = logs / "progress.jsonl"
                work = logs / "work"
                work.mkdir(parents=True, exist_ok=False)
                progress = directory / "progress.json"
                progress.unlink(missing_ok=True)
                job.update(
                    attempt=attempt,
                    work=str(work),
                    progress=str(progress.relative_to(run)),
                    progress_journal=str(attempt_journal.relative_to(run)),
                    state="running",
                )
                save_campaign(run / "campaign.json", state)
                old_bytes = sum(contained_path(root, path).stat().st_size
                                for path, value in job["previous"].items() if value)
                # Conservative planning estimate, not a promise of final JSON size.
                required = job["rows"] * 2048 + old_bytes + 1024**3
                if shutil.disk_usage(work).free < required:
                    raise RuntimeError(f"Insufficient disk margin: need about {required} bytes for this job")
                for item in job["inputs"]:
                    copy_frozen(contained_path(run / "inputs", item), contained_path(work, item))
                copy_frozen(
                    contained_path(run / "sources", job["recipe"]),
                    contained_path(work, job["recipe"]),
                )
                job["gpu_before"] = gpu_observation()
                save_campaign(run / "campaign.json", state)
                try:
                    job["generation_wall_seconds"] = run_generator(
                        binary,
                        work,
                        logs,
                        progress,
                        checkpoint,
                        job.get("checkpoint_id"),
                    )
                finally:
                    job["gpu_after"] = gpu_observation()
                for item in job["inputs"]:
                    if digest(contained_path(work, item)) != state["input_hashes"][item]:
                        raise ValueError(f"Generator modified a parameter input: {item}")
                started = time.perf_counter()
                job["artifacts"] = check_outputs(work, job)
                parity = compare_matching_prices(state, job, work)
                if parity is not None:
                    job["price_parity"] = parity
                    if parity["outside_budget_rows"]:
                        print(
                            f"Quality review: {job['target']} has "
                            f"{parity['outside_budget_rows']} central prices outside "
                            "the cross-kernel comparison budget",
                            flush=True,
                        )
                attach_generation(work, job, state)
                # The publication journal must cover the enriched YAML, not
                # the native intermediate. JSON bytes remain untouched.
                job["artifacts"][1]["sha256"] = digest(contained_path(work, job["generation"]))
                job["artifact_check_seconds"] = time.perf_counter() - started
                job["state"] = "staged"
                save_campaign(run / "campaign.json", state)
            elif job.get("progress_journal"):
                attempt_journal = contained_path(run, job["progress_journal"])
            if checkpoint is not None and checkpoint.exists():
                shutil.rmtree(checkpoint)
                job["checkpoint_cleared"] = True
                save_campaign(run / "campaign.json", state)
                if attempt_journal is not None:
                    append_journal(attempt_journal, "checkpoint_cleared")
            if state["publish"]:
                started = time.perf_counter()
                try:
                    publish_pair(root, Path(job["work"]), directory / "publication.json", job["artifacts"])
                finally:
                    job["publication_seconds"] = job.get("publication_seconds", 0) + time.perf_counter() - started
            job["state"] = "complete"
            job.pop("last_error", None)
            save_campaign(run / "campaign.json", state)
            if attempt_journal is not None:
                append_journal(attempt_journal, "job_complete", published=state["publish"])
        except BaseException as error:
            job["last_error"] = str(error) or type(error).__name__
            # A staged job retains its publication journal; do not regenerate it.
            if job["state"] != "staged":
                job["state"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            save_campaign(run / "campaign.json", state)
            if attempt_journal is not None:
                try:
                    append_journal(attempt_journal, "job_interrupted" if isinstance(error, KeyboardInterrupt)
                                   else "job_failed", error=job["last_error"])
                except OSError:
                    pass
            raise


def interrupt_campaign(*_arguments) -> None:
    raise KeyboardInterrupt("campaign interrupted; resume explicitly")



def amend_verifier(run: Path, state: dict, root: Path) -> None:
    """Record a verifier-only change while keeping frozen binaries and inputs.

    The original controller and source archive remain in the campaign. The
    replacement controller is copied separately and cited by later receipts.
    This is allowed only for a staged campaign, once, and only when all other
    controller files still match their frozen hashes.
    """
    name = "tools/datasets/generate_catalog.py"
    if state["publish"] or state.get("controller_amendments"):
        raise ValueError("Verifier amendment requires an unpublished, unamended campaign")
    old_hash = state["controller_hashes"][name]
    new_hash = digest(contained_path(root, name))
    if old_hash == new_hash or digest(contained_path(run / "sources", name)) != old_hash:
        raise ValueError("No valid frozen controller change to amend")
    for other, expected in state["controller_hashes"].items():
        if other != name and digest(contained_path(root, other)) != expected:
            raise ValueError(f"Other controller changed since this campaign was frozen: {other}")
    snapshot = f"amendments/generate_catalog-{new_hash}.py"
    if (run / snapshot).exists():
        raise ValueError("Verifier amendment snapshot already exists")
    if copy_frozen(contained_path(root, name), run / snapshot) != new_hash:
        raise ValueError("Verifier amendment copy changed")
    amendment = {
        "kind": "central_price_comparison_report_only",
        "controller": name,
        "previous_sha256": old_hash,
        "sha256": new_hash,
        "snapshot": snapshot,
        "unix_time": time.time(),
        "completed_jobs_before": sum(job["state"] == "complete" for job in state["jobs"]),
    }
    state["controller_amendments"] = [amendment]
    state["controller_hashes"][name] = new_hash
    save_campaign(run / "campaign.json", state)


def verify_verifier_amendment(run: Path, state: dict) -> None:
    for amendment in state.get("controller_amendments", []):
        name = amendment["controller"]
        if (digest(contained_path(run / "sources", name)) != amendment["previous_sha256"]
                or digest(contained_path(run, amendment["snapshot"])) != amendment["sha256"]
                or state["controller_hashes"][name] != amendment["sha256"]):
            raise ValueError("Verifier amendment provenance changed")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--kind", choices=("prices", "price_gradients", "samples", "all"), default="prices")
    parser.add_argument("--asset-class", action="append", choices=("equity", "fixed_income"), default=[])
    parser.add_argument("--model-family", action="append", choices=("markovian", "rough"), default=[])
    parser.add_argument("--construction", action="append", choices=("aligned", "cartesian"), default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument(
        "--input-override", action="append", default=[], metavar="LOGICAL_PATH=SOURCE_PATH",
        help="freeze SOURCE_PATH as a selected logical parameter input without editing the canonical dataset",
    )
    parser.add_argument("--skip-published", action="store_true",
                        help="exclude complete JSON/receipt pairs and reject partial pairs")
    parser.add_argument("--catalog-only", action="store_true",
                        help="select generators physically present in the published catalog only")
    parser.add_argument("--compile", action="store_true", help="build only the selected generators")
    parser.add_argument("--compile-jobs", type=int, default=1)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execute", action="store_true", help="otherwise only inspect the selection")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish immutable staged outputs; differing existing bytes are rejected",
    )
    parser.add_argument("--resume", action="store_true", help="explicitly resume the frozen campaign")
    parser.add_argument("--amend-verifier", action="store_true",
                        help="resume an unpublished run with this recorded parity-verifier update")
    arguments = parser.parse_args()
    if arguments.resume and (not arguments.execute or not arguments.run_dir):
        parser.error("--resume requires --execute and --run-dir")
    if arguments.amend_verifier and not arguments.resume:
        parser.error("--amend-verifier requires --resume")
    if arguments.publish and not arguments.execute:
        parser.error("--publish requires --execute")
    if arguments.compile_jobs <= 0:
        parser.error("--compile-jobs must be positive")
    if arguments.compile_jobs != 1 and not arguments.compile:
        parser.error("--compile-jobs requires --compile")
    if arguments.resume:
        state = json.loads((arguments.run_dir / "campaign.json").read_text())
        validate_document(state, "campaign", arguments.run_dir / "campaign.json")
        if state["version"] != 4 or state["root"] != str(ROOT):
            raise ValueError("Unsupported campaign version or repository; retain its original controller")
        changed_controllers = [name for name, expected in state["controller_hashes"].items()
                               if digest(contained_path(ROOT, name)) != expected]
        if changed_controllers != (["tools/datasets/generate_catalog.py"]
                                   if arguments.amend_verifier else []):
            raise ValueError("Controller changed since this campaign was frozen: "
                             + ", ".join(changed_controllers))
        if (arguments.model or arguments.target or arguments.asset_class
                or arguments.model_family or arguments.construction
                or arguments.input_override or arguments.skip_published
                or arguments.catalog_only or arguments.compile
                or arguments.kind != "prices"
                or (arguments.publish and not state["publish"])):
            raise ValueError("Resume uses the frozen selection and publication policy; do not override them")
        jobs = state["jobs"]
    else:
        kinds = {"prices", "price_gradients", "samples"} if arguments.kind == "all" else {arguments.kind}
        jobs = inventory(
            ROOT,
            kinds,
            set(arguments.model),
            set(arguments.target),
            asset_classes=set(arguments.asset_class),
            model_families=set(arguments.model_family),
            constructions=set(arguments.construction),
            skip_published=arguments.skip_published,
            catalog_only=arguments.catalog_only,
        )
    build = Path(state["build"]) if arguments.resume else arguments.build.resolve()
    if not jobs:
        print(json.dumps({"job_count": 0, "jobs": []}, indent=2))
        return 0
    if arguments.compile:
        compile_selected(ROOT, build, jobs, arguments.compile_jobs)
    if not arguments.execute:
        print(json.dumps({"job_count": len(jobs), "jobs": jobs}, indent=2))
        return 0
    if arguments.run_dir is None:
        parser.error("--execute requires a new --run-dir (or --resume)")
    overrides = {}
    for assignment in arguments.input_override:
        logical, separator, source = assignment.partition("=")
        if not separator or not logical.startswith("datasets/") or not source:
            parser.error("--input-override requires datasets/...json=SOURCE_PATH")
        if logical in overrides:
            parser.error(f"Duplicate input override: {logical}")
        overrides[logical] = Path(source).resolve()
    # One lock for the repository, including controllers using different builds.
    lock_path = ROOT / "work" / "generation" / ".campaign.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = arguments.run_dir.resolve()
        if not arguments.resume:
            state = freeze(ROOT, build, run, jobs, arguments.publish, overrides)
        elif arguments.amend_verifier:
            amend_verifier(run, state, ROOT)
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
