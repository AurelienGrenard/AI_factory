"""Reject gradient artifacts that disagree with frozen selection, stencils or execution."""

import math
import struct


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def check_outputs(job, catalog, document):
    parameters = job["sensitivity"]["parameters"]
    names = [item["parameter"] for item in parameters]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate selected gradient parameter")
    paths = job["launch_plan"]["paths_per_price"]
    for artifact in (catalog, document):
        if (artifact.get("sensitivity") != job["sensitivity"]
                or artifact.get(job["time_key"]) != job["time_configuration"]):
            raise ValueError("Gradient sensitivity/time grid contradicts frozen recipe")
        execution = artifact.get("summary", {})
        for key in ("paths_per_price", "threads_per_block", "sensitivity_count", "scenario_count", "gradient_layout",
                    "prices_per_launch", "block_count", "sensitivity_batch_size", "sensitivity_batches_per_price",
                    "maximum_live_scenarios", "kernel_launches_per_price_batch", "full_batch_block_count",
                    "tail_batch_block_count", "work_distribution", "price_moment_batches_per_price", "central_work_policy"):
            if key in job["launch_plan"] and execution.get(key) != job["launch_plan"][key]:
                raise ValueError("Gradient execution contradicts compiled plan: " + key)
        if paths and execution.get("seed") != job["rng_stream_seeds"]["dynamics"]:
            raise ValueError("Gradient seed contradicts frozen CRN reservation")
    for row in document["results"]:
        outputs = row["outputs"]
        if list(outputs.get("gradients", {})) != names or list(row.get("stencils", {})) != names:
            raise ValueError("Gradient columns disagree with selection order")
        if paths:
            errors = outputs.get("gradient_standard_errors", {})
            if list(errors) != names or any(type(value) not in (int,float) or value < 0 or not math.isfinite(value)
                                           for value in errors.values()):
                raise ValueError("Missing or invalid paired gradient errors")
        elif "gradient_standard_errors" in outputs or "standard_error" in outputs:
            raise ValueError("Closed-form gradients must not publish sampling errors")
        for selection in parameters:
            name = selection["parameter"]
            value = outputs["gradients"][name]
            if type(value) not in (int,float) or not math.isfinite(value):
                raise ValueError("Invalid gradient value")
            stencil = row["stencils"][name]
            x, first, second = (stencil[key] for key in ("central","first","second"))
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,first,second)):
                raise ValueError("Invalid stencil endpoints")
            kind = stencil["kind"]
            ordered = {"centered": first < x < second, "forward": x < first < second, "backward": second < first < x}
            if not ordered.get(kind, False) or stencil["represented_width"] != fp32(second-first):
                raise ValueError("Invalid represented stencil orientation/width")
            requested = fp32(selection["displacement"])
            if selection["scale"] == "relative":
                requested = fp32(requested*abs(x))
            elif selection["scale"] != "absolute":
                raise ValueError("Unknown bump scale")
            if requested <= 0 or stencil["displacement"] != requested:
                raise ValueError("Stencil displacement contradicts requested bump")
            offsets = {"centered":(-1,1), "forward":(1,2), "backward":(-1,-2)}[kind]
            if name == "product.maturity_years":
                steps_per_year = (job["time_configuration"].get("steps_per_year")
                    or job["time_configuration"].get("maturity_bump_steps_per_year"))
                dt = fp32(1/steps_per_year)
                steps, bump_steps = round(x/dt), round(requested/dt)
                if bump_steps < 1 or fp32(bump_steps*dt) != requested or fp32(steps*dt) != x:
                    raise ValueError("Maturity stencil is not on the declared integer grid")
                endpoints = tuple(fp32((steps+offset*bump_steps)*dt) for offset in offsets)
            else:
                endpoints = tuple(fp32(x+offset*requested) for offset in offsets)
            if (first,second) != endpoints:
                raise ValueError("Stencil endpoints contradict requested displacement")
            if selection["boundary"] == "central_only" and kind != "centered":
                raise ValueError("A centered-only sensitivity was silently shifted")
            if kind == "centered":
                expected_weights = (0.,0.)
            else:
                a,b = first-x,second-x
                expected_weights = (fp32(b/(a*(b-a))),fp32(-a/(b*(b-a))))
            if (stencil["first_weight"],stencil["second_weight"]) != expected_weights:
                raise ValueError("Stencil weights disagree with represented spacing")
