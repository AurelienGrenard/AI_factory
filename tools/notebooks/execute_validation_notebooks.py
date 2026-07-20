"""Execute every validation notebook in a fresh Jupyter kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "validation"


def execute_notebook(path: Path, timeout: int) -> None:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    paths = sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(PROJECT_ROOT)
        print(f"[{index:02d}/{len(paths):02d}] {relative}", flush=True)
        execute_notebook(path, args.timeout)

    print(f"Executed {len(paths)} validation notebooks.", flush=True)


if __name__ == "__main__":
    main()
