"""Display a catalogue campaign's progress without changing its generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REFRESH_SECONDS = 10
FINAL_STATES = {"complete", "failed", "interrupted"}
STATE_LABELS = {
    "pending": "en attente",
    "running": "calcul en cours",
    "staged": "publication en cours",
    "complete": "terminée",
    "failed": "échec",
    "interrupted": "interrompue",
}


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "en attente du premier relevé"
    hours, remainder = divmod(max(0, round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours} h {minutes:02d} min {seconds:02d} s"


def number(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def decimal(value: float, digits: int) -> str:
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def select_job(campaign: dict, target: str | None) -> dict:
    jobs = campaign["jobs"]
    if target is not None:
        for job in jobs:
            if job["target"] == target:
                return job
        raise ValueError(f"Cible absente de cette campagne : {target}")
    for job in jobs:
        if job["state"] in {"running", "staged"}:
            return job
    for job in jobs:
        if job["state"] != "complete":
            return job
    return jobs[-1]


def read_progress(run: Path, job: dict) -> dict:
    relative = job.get("progress")
    if not relative:
        return {}
    try:
        return json.loads((run / relative).read_text())
    except FileNotFoundError:
        return {}


def format_progress(job: dict, progress: dict, now: float, campaign: dict | None = None) -> str:
    state = job["state"]
    label = STATE_LABELS.get(state, state)
    if state == "running" and progress.get("state") == "complete":
        label = "calcul terminé, finalisation en cours"
    sample_job = job.get("kind") == "samples"
    total_key = "total_samples" if sample_job else "total_prices"
    completed_key = "completed_samples" if sample_job else "completed_prices"
    rate_key = "samples_per_second" if sample_job else "prices_per_second"
    total = int(progress.get(total_key, job.get("rows", 0)))
    completed = int(progress.get(completed_key, 0))
    percent = 100 * completed / total if total else 0.0
    elapsed = progress.get("elapsed_seconds")
    rate = progress.get(rate_key, 0.0)
    remaining = progress.get("estimated_seconds_remaining")

    lines = []
    if campaign is not None:
        jobs = campaign["jobs"]
        finished = sum(item["state"] == "complete" for item in jobs)
        position = next(index for index, item in enumerate(jobs, 1)
                        if item["target"] == job["target"])
        lines.extend((
            f"Campagne : {finished} / {len(jobs)} datasets terminés "
            f"({decimal(100 * finished / len(jobs), 2)} %)",
            f"Dataset  : {position} / {len(jobs)} — {Path(job['dataset']).stem}",
        ))
    else:
        lines.append(f"Dataset : {Path(job['dataset']).stem}")
    lines.extend((f"État     : {label}", ""))
    if sample_job and not progress:
        lines.extend((
            "Échantillons écrits : en attente de la phase d'écriture",
            "Avancement         : indisponible pendant la simulation",
            "Débit              : en attente",
            "Temps écoulé       : en attente du premier relevé",
        ))
    elif sample_job:
        lines.extend((
            f"Échantillons écrits : {number(completed)} / {number(total)}",
            f"Avancement         : {decimal(percent, 2)} % de l'écriture",
            f"Débit              : {decimal(rate, 1)} échantillons/s",
            f"Temps écoulé       : {duration(elapsed)} depuis le début de l'écriture",
        ))
    else:
        lines.extend((
            f"Prix calculés : {number(completed)} / {number(total)}",
            f"Avancement   : {decimal(percent, 2)} %",
            f"Débit      : {decimal(rate, 1)} prix/s" if progress else "Débit      : en attente",
            f"Temps écoulé : {duration(elapsed) if elapsed is not None else 'en attente'}",
        ))
    if progress.get("state") == "complete":
        eta = "calcul terminé"
    elif state in {"failed", "interrupted"}:
        eta = "indisponible"
    else:
        eta = duration(remaining)
    scope = "écriture JSON" if sample_job else "dataset"
    lines.append(f"Temps restant estimé ({scope}) : {eta}")
    timestamp = progress.get("unix_time")
    if timestamp is not None:
        lines.extend(("", f"Dernier relevé : il y a {duration(now - timestamp)}"))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="dossier de campagne contenant campaign.json")
    parser.add_argument("--target", help="cible à suivre dans une campagne à plusieurs jobs")
    parser.add_argument("--once", action="store_true", help="afficher un seul relevé")
    args = parser.parse_args()

    run = args.run_dir.resolve()
    if not (run / "campaign.json").is_file():
        parser.error(f"Campagne introuvable : {run / 'campaign.json'}")
    try:
        while True:
            campaign = json.loads((run / "campaign.json").read_text())
            job = select_job(campaign, args.target)
            progress = read_progress(run, job)
            if sys.stdout.isatty() and not args.once:
                print("\033[2J\033[H", end="")
            print(format_progress(job, progress, time.time(), campaign), flush=True)
            if args.once or job["state"] in FINAL_STATES:
                return 0
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\nSuivi arrêté ; la génération continue.")
        return 0
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(1, f"Impossible de lire l'avancement : {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
