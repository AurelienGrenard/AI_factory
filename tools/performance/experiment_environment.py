"""Separate experiment execution guards from power-related timing eligibility.

Power limits and temperatures are observations, never reasons to kill a long
experiment. Hardware power brake, external power and concurrency guards remain
distinct. Official regression-baseline admission is not changed by this module.
"""
from __future__ import annotations

import math

try:
    from .run_baseline import validate_preflight
except ImportError:
    from run_baseline import validate_preflight


def power_comparability_issue(current: float | None, reference: float | None,
                              minimum: float | None = None, maximum: float | None = None,
                              relative_tolerance: float = .10) -> str | None:
    """Describe a timing exclusion without stopping the process or changing power."""
    if any(value is None or not math.isfinite(value) or value <= 0
           for value in (current, reference)):
        return "GPU power-limit telemetry is unavailable or invalid"
    if minimum is not None and current < minimum:
        return "GPU power limit is below the timing-comparison bound"
    if maximum is not None and current > maximum + .1:
        return "GPU power limit is above the timing-comparison bound"
    if abs(current / reference - 1) > relative_tolerance:
        return f"GPU power limit changed by more than {100 * relative_tolerance:g}%"
    return None


def record_timing_issue(outcome: dict, issue: str | None) -> None:
    """Make exclusions sticky, including excursions that later return to normal."""
    if issue is not None:
        reasons = outcome.setdefault("timing_ineligibility_reasons", [])
        if issue not in reasons:
            reasons.append(issue)
        outcome["timing_eligible"] = False


def hardware_power_brake_issue(throttle: dict) -> str | None:
    if any(value == "Active" and "hw_power_brake" in name
           for name, value in throttle.items()):
        return "GPU hardware power brake"
    return None


def validate_experiment_preflight(profile: dict, snapshot: dict) -> None:
    """Retain identity/AC/concurrency/brake guards, not a GPU power-limit veto."""
    policy = dict(profile["decision_policy"]["preflight"])
    policy.pop("minimum_current_power_limit_w", None)
    validate_preflight({**profile, "decision_policy": {
        **profile["decision_policy"], "preflight": policy}}, snapshot)
