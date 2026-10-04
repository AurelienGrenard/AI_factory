#!/usr/bin/env python3
"""Build the website data exclusively from the portable prod bundle."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "catalog" / "prod" / "manifest.json"
OUTPUT = ROOT / "AI_factory_website" / "static" / "dataset-index.js"
VISIBLE = {"samples", "model_parameters", "prices"}


def build() -> None:
    manifest = json.loads(MANIFEST.read_text())
    rows = [row for row in manifest["datasets"] if row["kind"] in VISIBLE]
    if any(not row["catalogPath"].startswith("catalog/prod/")
           or not row["output"].startswith("datasets/prod/") for row in rows):
        raise ValueError("The website index must contain only prod paths")
    if any(not row["ready"] for row in rows):
        raise ValueError("The prod bundle contains an unpublished visible dataset")
    keys = set()
    for row in rows:
        group = (row["model"], row["kind"], row["product"] if row["kind"] == "prices" else None)
        if group in keys:
            raise ValueError(f"Several prod datasets at one website destination: {group}")
        keys.add(group)
    OUTPUT.write_text("window.AI_FACTORY_DATASET_INDEX = " + json.dumps(
        {"count": len(rows), "datasets": rows}, separators=(",", ":"), ensure_ascii=False
    ) + ";\n")
    print(f"Wrote {len(rows)} visible datasets to {OUTPUT}")


if __name__ == "__main__":
    build()
