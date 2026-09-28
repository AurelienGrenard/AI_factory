"""Display one terminal-GAN run without changing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


FINAL_STATES = {"completed", "failed", "interrupted", "invalid"}


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "indisponible"
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours} h {minutes:02d} min {seconds:02d} s"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_run_state(directory: Path, state: dict) -> list[str]:
    lines = [
        f"Run      : {directory.name}",
        f"Etat     : {state.get('status', 'inconnu')}",
        (
            "Epoques  : "
            f"{state.get('completed_epochs', 0)} / "
            f"{state.get('total_epochs', '?')}"
        ),
    ]
    if "current_epoch" in state:
        lines.append(
            "Batch    : "
            f"{state.get('completed_batches', 0)} / "
            f"{state.get('total_batches', '?')} "
            f"(epoque {state['current_epoch']})"
        )
    lines.extend(
        (
            f"G steps  : {state.get('generator_steps', 0)}",
            f"C steps  : {state.get('critic_steps', 0)}",
            (
                "Ecoule   : "
                f"{_duration(state.get('elapsed_seconds'))}"
            ),
            (
                "Restant  : "
                f"{_duration(state.get('estimated_seconds_remaining'))}"
            ),
        )
    )
    recent = state.get("recent")
    if isinstance(recent, dict):
        lines.extend(
            (
                "",
                f"G loss   : {recent.get('generator_loss', float('nan')):.6g}",
                f"C loss   : {recent.get('critic_loss', float('nan')):.6g}",
                f"Wasserst.: {recent.get('wasserstein', float('nan')):.6g}",
                f"GP       : {recent.get('gradient_penalty', float('nan')):.6g}",
                f"|grad C| : {recent.get('gradient_norm_mean', float('nan')):.6g}",
            )
        )
        if "mode_seeking_ratio" in recent:
            lines.append(
                "Diversite: %.6g"
                % float(recent["mode_seeking_ratio"])
            )
        if "moment_matching" in recent:
            lines.append(
                "Moments  : %.6g"
                % float(recent["moment_matching"])
            )
    if state.get("error"):
        lines.extend(("", f"Erreur   : {state['error']}"))
    return lines


def _format(directory: Path) -> tuple[str, str]:
    state_path = directory / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"No state.json in {directory}")
    state = _read(state_path)
    return "\n".join(format_run_state(directory, state)), str(
        state.get("status", "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--refresh", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.refresh <= 0.0:
        parser.error("--refresh must be positive")
    directory = arguments.directory.resolve()
    try:
        while True:
            display, status = _format(directory)
            if sys.stdout.isatty() and not arguments.once:
                print("\033[2J\033[H", end="")
            print(display, flush=True)
            if arguments.once or status in FINAL_STATES:
                return 0
            time.sleep(arguments.refresh)
    except KeyboardInterrupt:
        print("\nSuivi arrete ; l'entrainement continue.")
        return 0
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.exit(
            1, f"Impossible de lire l'avancement : {error}\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
