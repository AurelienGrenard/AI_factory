"""Reject sensitivity artifacts that contradict frozen outputs or stencils."""

import math
import struct


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


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


def _first_support(stencil):
    if stencil["kind"] == "centered":
        width = stencil["represented_width"]
        return [1, 2], [fp32(-1.0 / width), fp32(1.0 / width)]
    return (
        [0, 1, 2],
        [
            fp32(-(stencil["first_weight"] + stencil["second_weight"])),
            stencil["first_weight"],
            stencil["second_weight"],
        ],
    )


def _check_mixed_stencil(stencil, first_axis, second_axis):
    first_nodes, first_weights = _first_support(first_axis)
    second_nodes, second_weights = _first_support(second_axis)
    expected_first = []
    expected_second = []
    expected_weights = []
    for first_index, first_weight in zip(first_nodes, first_weights):
        for second_index, second_weight in zip(
            second_nodes, second_weights
        ):
            expected_first.append(first_index)
            expected_second.append(second_index)
            expected_weights.append(fp32(first_weight * second_weight))
    if (
        stencil.get("node_count") != len(expected_weights)
        or stencil.get("first_local_nodes") != expected_first
        or stencil.get("second_local_nodes") != expected_second
        or stencil.get("weights") != expected_weights
    ):
        raise ValueError("Mixed stencil disagrees with its axis stencils")


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
    axis_name_set = set(first_names) | set(diagonal_names)
    for first, second in mixed_pairs:
        axis_name_set.update((first, second))
    axis_names = [name for name in names if name in axis_name_set]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate selected gradient parameter")

    paths = job["launch_plan"]["paths_per_price"]
    for artifact in (catalog, document):
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

    for row in document["results"]:
        outputs = row["outputs"]
        stencils = row.get("stencils", {})
        if (
            (
                first_requested
                and list(outputs.get("gradients", {})) != first_names
            )
            or (not first_requested and "gradients" in outputs)
            or (
                second_requested
                and list(outputs.get("diagonal_hessians", {}))
                    != diagonal_names
            )
            or (
                not second_requested
                and "diagonal_hessians" in outputs
            )
            or (
                mixed_requested
                and list(outputs.get("mixed_hessians", {})) != mixed_names
            )
            or (not mixed_requested and "mixed_hessians" in outputs)
            or list(stencils) != axis_names
            or (
                mixed_requested
                and list(row.get("mixed_stencils", {})) != mixed_names
            )
            or (not mixed_requested and "mixed_stencils" in row)
        ):
            raise ValueError("Gradient columns disagree with selection order")

        if paths:
            requested_errors = (
                (
                    first_requested,
                    "gradient_standard_errors",
                    first_names,
                ),
                (
                    second_requested,
                    "diagonal_hessian_standard_errors",
                    diagonal_names,
                ),
                (
                    mixed_requested,
                    "mixed_hessian_standard_errors",
                    mixed_names,
                ),
            )
            for requested, key, expected in requested_errors:
                errors = outputs.get(key, {})
                if (
                    (requested and list(errors) != expected)
                    or (not requested and key in outputs)
                    or any(
                        type(value) not in (int, float)
                        or value < 0
                        or not math.isfinite(value)
                        for value in errors.values()
                    )
                ):
                    raise ValueError(
                        "Missing or invalid paired sensitivity errors"
                    )
        elif any(
            key in outputs
            for key in (
                "gradient_standard_errors",
                "diagonal_hessian_standard_errors",
                "mixed_hessian_standard_errors",
                "standard_error",
            )
        ):
            raise ValueError(
                "Closed-form gradients must not publish sampling errors"
            )

        for selection in parameters:
            name = selection["parameter"]
            for key, selected_names in (
                ("gradients", first_names),
                ("diagonal_hessians", diagonal_names),
            ):
                if name in selected_names:
                    value = outputs[key][name]
                    if (
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                    ):
                        raise ValueError("Invalid sensitivity value")

            if name not in axis_name_set:
                continue
            stencil = stencils[name]
            x, first, second = (
                stencil[key] for key in ("central", "first", "second")
            )
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (x, first, second)
            ):
                raise ValueError("Invalid stencil endpoints")
            kind = stencil["kind"]
            ordered = {
                "centered": first < x < second,
                "forward": x < first < second,
                "backward": second < first < x,
            }
            if (
                not ordered.get(kind, False)
                or stencil["represented_width"] != fp32(second - first)
            ):
                raise ValueError(
                    "Invalid represented stencil orientation/width"
                )
            requested = fp32(selection["displacement"])
            if selection["scale"] == "relative":
                requested = fp32(requested * abs(x))
            elif selection["scale"] != "absolute":
                raise ValueError("Unknown bump scale")
            if requested <= 0 or stencil["displacement"] != requested:
                raise ValueError(
                    "Stencil displacement contradicts requested bump"
                )
            offsets = {
                "centered": (-1, 1),
                "forward": (1, 2),
                "backward": (-1, -2),
            }[kind]
            if name == "product.maturity_years":
                steps_per_year = (
                    job["time_configuration"].get("steps_per_year")
                    or job["time_configuration"].get(
                        "maturity_bump_steps_per_year"
                    )
                )
                dt = fp32(1 / steps_per_year)
                steps = round(x / dt)
                bump_steps = round(requested / dt)
                if (
                    bump_steps < 1
                    or fp32(bump_steps * dt) != requested
                    or fp32(steps * dt) != x
                ):
                    raise ValueError(
                        "Maturity stencil is not on the declared integer grid"
                    )
                endpoints = tuple(
                    fp32((steps + offset * bump_steps) * dt)
                    for offset in offsets
                )
            else:
                endpoints = tuple(
                    fp32(x + offset * requested) for offset in offsets
                )
            if (first, second) != endpoints:
                raise ValueError(
                    "Stencil endpoints contradict requested displacement"
                )
            if (
                selection["boundary"] == "central_only"
                and kind != "centered"
            ):
                raise ValueError(
                    "A centered-only sensitivity was silently shifted"
                )
            if kind == "centered":
                expected_weights = (0.0, 0.0)
            else:
                first_offset, second_offset = first - x, second - x
                expected_weights = (
                    fp32(
                        second_offset
                        / (
                            first_offset
                            * (second_offset - first_offset)
                        )
                    ),
                    fp32(
                        -first_offset
                        / (
                            second_offset
                            * (second_offset - first_offset)
                        )
                    ),
                )
            if (
                stencil["first_weight"],
                stencil["second_weight"],
            ) != expected_weights:
                raise ValueError(
                    "Stencil weights disagree with represented spacing"
                )

            if name in diagonal_names:
                node_count = 3 if kind == "centered" else 4
                weights = stencil.get("second_weights")
                if (
                    stencil.get("node_count") != node_count
                    or not isinstance(weights, list)
                    or len(weights) != node_count
                ):
                    raise ValueError("Incomplete diagonal stencil")
                nodes = [x, first, second]
                if node_count == 4:
                    third = stencil.get("third")
                    if (
                        type(third) not in (int, float)
                        or not math.isfinite(third)
                    ):
                        raise ValueError("Invalid third diagonal node")
                    if name == "product.maturity_years":
                        expected_third = fp32(
                            (
                                steps
                                + (
                                    3 if kind == "forward" else -3
                                )
                                * bump_steps
                            )
                            * dt
                        )
                    else:
                        expected_third = fp32(
                            x
                            + (3 if kind == "forward" else -3)
                            * requested
                        )
                    if third != expected_third:
                        raise ValueError(
                            "Third diagonal node contradicts the bump"
                        )
                    nodes.append(third)
                elif "third" in stencil:
                    raise ValueError(
                        "Centered diagonal stencil has a third node"
                    )
                if any(
                    type(weight) not in (int, float)
                    or not math.isfinite(weight)
                    for weight in weights
                ):
                    raise ValueError("Invalid diagonal weights")
                quadratic = sum(
                    weight * (node - x) ** 2
                    for weight, node in zip(weights, nodes)
                )
                if not math.isclose(
                    quadratic, 2.0, rel_tol=0.01, abs_tol=0.01
                ):
                    raise ValueError(
                        "Diagonal weights do not reconstruct a quadratic"
                    )

        if mixed_requested:
            for label, (first_name, second_name) in zip(
                mixed_names, mixed_pairs
            ):
                value = outputs["mixed_hessians"][label]
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                ):
                    raise ValueError("Invalid mixed sensitivity value")
                _check_mixed_stencil(
                    row["mixed_stencils"][label],
                    stencils[first_name],
                    stencils[second_name],
                )
