"""Validate compact price-gradient rows against their frozen recipe and launch plan."""

import math


def _mixed_pairs(sensitivity, names):
    declaration = sensitivity.get("mixed_second")
    if declaration == "all":
        return [
            (names[first], names[second])
            for first in range(len(names))
            for second in range(first + 1, len(names))
        ]
    if isinstance(declaration, list):
        pairs = []
        for pair in declaration:
            if not isinstance(pair, dict):
                raise ValueError("Invalid mixed sensitivity pair")
            first, second = pair.get("first"), pair.get("second")
            if first not in names or second not in names or first == second:
                raise ValueError("Invalid mixed sensitivity pair")
            canonical = (
                (first, second)
                if names.index(first) < names.index(second)
                else (second, first)
            )
            if canonical in pairs:
                raise ValueError("Duplicate mixed sensitivity pair")
            pairs.append(canonical)
        return pairs
    raise ValueError("Mixed derivative order lacks its pair selection")


def _selected_names(sensitivity, key, names, requested):
    declaration = sensitivity.get(key)
    if not requested:
        if declaration is not None:
            raise ValueError("Unrequested sensitivity order has a selection")
        return []
    if declaration is None or declaration == "all":
        return names
    if (
        not isinstance(declaration, list)
        or any(name not in names for name in declaration)
        or len(set(declaration)) != len(declaration)
    ):
        raise ValueError("Invalid selected sensitivity coordinates")
    return declaration


def _check_compact_outputs(document, first_names, diagonal_names,
                           mixed_names, paths, has_curve, option_side):
    header = {"database_id", "model_family", "product_family", "catalog", "row_count"}
    if option_side is not None:
        if option_side not in {"call", "put", "payer", "receiver"}:
            raise ValueError("Invalid frozen option side")
        header.add("option_side")
    references = {"model_dataset", "product_dataset"}
    if has_curve:
        header.add("curve_family")
        references.add("curve_dataset")
    if (not isinstance(document, dict)
            or set(document) != header | references | {"results"}):
        raise ValueError("Gradient JSON must contain identity, input references and results only")
    if any(not isinstance(document[key], str) or not document[key]
           for key in header - {"row_count"}):
        raise ValueError("Invalid gradient dataset identity")
    if option_side is not None and document["option_side"] != option_side:
        raise ValueError("Gradient option side contradicts frozen price variant")
    for key in references:
        reference = document[key]
        if (not isinstance(reference, dict)
                or set(reference) != {"id", "catalog"}
                or any(not isinstance(value, str) or not value
                       for value in reference.values())):
            raise ValueError(f"Invalid gradient {key} reference")
    if (type(document["row_count"]) is not int
            or document["row_count"] < 1
            or not isinstance(document["results"], list)
            or len(document["results"]) != document["row_count"]):
        raise ValueError("Gradient row count contradicts results")
    columns = {
        "gradients": first_names,
        "diagonal_hessians": diagonal_names,
        "mixed_hessians": mixed_names,
    }
    errors = {
        "gradient_standard_errors": first_names,
        "diagonal_hessian_standard_errors": diagonal_names,
        "mixed_hessian_standard_errors": mixed_names,
    }
    expected = {"model_id", "product_id", "price"}
    if has_curve:
        expected.add("curve_id")
    expected.update(key for key, names in columns.items() if names)
    if paths:
        expected.add("standard_error")
        expected.update(key for key, names in errors.items() if names)
    for index, row in enumerate(document["results"], 1):
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError(f"Invalid compact gradient columns at row {index}")
        for key in ("model_id", "product_id", "curve_id"):
            if key in expected and (not isinstance(row[key], str) or not row[key]):
                raise ValueError(f"Invalid {key} at row {index}")
        for key in ("price", "standard_error"):
            if key in expected:
                value = row[key]
                if (type(value) not in (int, float)
                        or not math.isfinite(value)
                        or (key == "standard_error" and value < 0)):
                    raise ValueError(f"Invalid {key} at row {index}")
        for key, names in {**columns, **errors}.items():
            if names:
                if key not in expected:
                    continue
                values = row[key]
                if not isinstance(values, dict) or list(values) != names:
                    raise ValueError(f"Invalid {key} selection at row {index}")
                if any(type(value) not in (int, float)
                       or not math.isfinite(value)
                       or (key in errors and value < 0)
                       for value in values.values()):
                    raise ValueError(f"Invalid {key} value at row {index}")


def check_outputs(job, catalog, document):
    parameters = job["sensitivity"]["parameters"]
    names = [item["parameter"] for item in parameters]
    orders = job["sensitivity"].get("orders", ["first"])
    valid_orders = {
        ("first",),
        ("diagonal_second",),
        ("first", "diagonal_second"),
        ("mixed_second",),
        ("first", "mixed_second"),
        ("diagonal_second", "mixed_second"),
        ("first", "diagonal_second", "mixed_second"),
    }
    if tuple(orders) not in valid_orders:
        raise ValueError("Unknown gradient derivative orders")
    first_requested = "first" in orders
    second_requested = "diagonal_second" in orders
    mixed_requested = "mixed_second" in orders
    if not mixed_requested and "mixed_second" in job["sensitivity"]:
        raise ValueError("Unrequested mixed order has a pair selection")
    first_names = _selected_names(
        job["sensitivity"], "first", names, first_requested
    )
    reconstructed = job["sensitivity"].get("reconstructible_from_price", [])
    if reconstructed:
        if not first_requested or not isinstance(reconstructed, list):
            raise ValueError("Analytic gradients require first-order output")
        if any(not isinstance(item, dict)
               or not all(isinstance(item.get(key), str) and item[key]
                          for key in ("parameter", "formula", "standard_error"))
               for item in reconstructed):
            raise ValueError("Invalid analytic gradient declaration")
        derived_names = [item["parameter"] for item in reconstructed]
        if len(set(derived_names)) != len(derived_names) or set(derived_names) & set(names):
            raise ValueError("Duplicate analytic gradient declaration")
        first_names += derived_names
    diagonal_names = _selected_names(
        job["sensitivity"],
        "diagonal_second",
        names,
        second_requested,
    )
    mixed_pairs = (
        _mixed_pairs(job["sensitivity"], names)
        if mixed_requested
        else []
    )
    mixed_names = [first + "|" + second for first, second in mixed_pairs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate selected gradient parameter")

    paths = job["launch_plan"]["paths_per_price"]
    for artifact in (catalog,):
        if (
            artifact.get("sensitivity") != job["sensitivity"]
            or artifact.get(job["time_key"]) != job["time_configuration"]
            or artifact.get("exercise_replay")
                != job.get("exercise_replay")
        ):
            raise ValueError(
                "Gradient sensitivity/time grid/replay contradicts frozen recipe"
            )
        execution = artifact.get("summary", {})
        for key in (
            "paths_per_price",
            "threads_per_block",
            "sensitivity_count",
            "scenario_count",
            "materialized_scenario_count",
            "gradient_layout",
            "prices_per_launch",
            "block_count",
            "sensitivity_batch_size",
            "sensitivity_batches_per_price",
            "maximum_live_scenarios",
            "represented_nodes_per_sensitivity",
            "requested_orders",
            "kernel_launches_per_price_batch",
            "sensitivity_block_count",
            "full_batch_block_count",
            "tail_batch_block_count",
            "work_distribution",
            "price_moment_batches_per_price",
            "central_work_policy",
            "sensitivity_graph_node_capacity",
            "first_sensitivity_count",
            "diagonal_hessian_count",
            "mixed_hessian_count",
        ):
            if (
                key in job["launch_plan"]
                and execution.get(key) != job["launch_plan"][key]
            ):
                raise ValueError(
                    "Gradient execution contradicts compiled plan: " + key
                )
        if (
            paths
            and execution.get("seed")
            != job["rng_stream_seeds"]["dynamics"]
        ):
            raise ValueError("Gradient seed contradicts frozen CRN reservation")

    _check_compact_outputs(
        document, first_names, diagonal_names, mixed_names, paths,
        job.get("declared_method", {}).get("has_curve", False),
        catalog.get("option_side"),
    )
