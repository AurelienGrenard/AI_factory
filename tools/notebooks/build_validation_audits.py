"""Normalize every validation notebook around one certification report."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "validation"


def _product_family(path: Path) -> str:
    stem = path.stem
    aliases = {
        "asian_arithmetic_calls": "asian_arithmetic",
        "lookback_fixed_calls": "lookback_fixed",
        "volatility_swaps": "volatility_swap",
    }
    candidates = sorted(
        (
            directory.name
            for directory in (PROJECT_ROOT / "registry/production/products").iterdir()
            if directory.is_dir()
        ),
        key=len,
        reverse=True,
    )
    matches = [
        candidate
        for candidate in candidates
        if any(
            f"_{token}_" in stem
            for token in {candidate, aliases.get(candidate, candidate)}
        )
    ]
    if not matches:
        raise ValueError(f"Cannot infer one product family from {path}.")
    return matches[0]


def _path_code(notebook: nbf.NotebookNode) -> str | None:
    for index, cell in enumerate(notebook.cells):
        source = cell.source
        if cell.cell_type == "code" and (
            "tools.paths" in source or "reconstruct_paths" in source
        ):
            return source
        if (
            cell.cell_type == "markdown"
            and "Path" in source
            and index + 1 < len(notebook.cells)
            and notebook.cells[index + 1].cell_type == "code"
        ):
            return notebook.cells[index + 1].source
    return None


def _regeneration_markdown(notebook: nbf.NotebookNode) -> str:
    start = next(
        (
            index for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "markdown" and "Regeneration Commands" in cell.source
        ),
        None,
    )
    if start is None:
        raise ValueError("Validation notebook has no regeneration section.")
    chunks = [
        cell.source.strip()
        for cell in notebook.cells[start:]
        if cell.cell_type == "markdown" and cell.source.strip()
    ]
    first = re.sub(
        r"^#{1,6}\s+Regeneration Commands\s*", "", chunks[0], count=1
    ).strip()
    body = [first, *chunks[1:]] if first else chunks[1:]
    return "## Regeneration Commands\n\n" + "\n\n".join(body)


def _setup_code(model: str, product: str, delta_crn: bool) -> str:
    return f'''from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from IPython.display import display

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "registry").is_dir())
PROJECT_ROOT = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src_python") not in sys.path:
    sys.path.insert(0, str(ROOT / "src_python"))

from tools.validation.audit import load_production_audit, timing_frame

pd.set_option("display.precision", 4)
audit = load_production_audit(
    ROOT,
    model_family={model!r},
    product_family={product!r},
    delta_crn={delta_crn!r},
)
cpp_gpu = audit.validation["cpp_gpu"].data
python_gpu = audit.validation["python_gpu"].data
cpp_cpu = audit.validation["cpp_cpu"].data
python_cpu = audit.validation["python_cpu"].data
cpp_gpu_spec = audit.validation["cpp_gpu"].specification
cpp_gpu_spec_summary = cpp_gpu_spec["summary"]
production = audit.production.data
production_head = production["results"][:100]
data = {{key: document.data for key, document in audit.validation.items()}}
specs = {{key: document.specification for key, document in audit.validation.items()}}
paths = {{key: document.json_path for key, document in audit.validation.items()}}
spec_paths = {{key: document.yaml_path for key, document in audit.validation.items()}}
spec_path = spec_paths["cpp_gpu"]
result_path = paths["cpp_gpu"]
PRODUCT_FAMILY = {product!r}
performance = timing_frame(audit)
benchmark_row_count = performance.attrs.get("benchmark_row_count")
if benchmark_row_count and benchmark_row_count != 100:
    print(
        f"Performance timing uses one hot call over {{benchmark_row_count:,}} "
        "benchmark rows; correctness retains the 100-row audit slice."
    )
display(performance.style.format("{{:.4g}}", na_rep="-"))
ax = performance["wall seconds"].plot(kind="bar", figsize=(7, 3), rot=0)
ax.set_ylabel("wall seconds")
ax.set_title("Execution timing")
plt.show()'''


COHERENCE_CODE = '''from tools.validation.audit import coherence_frame

coherence = coherence_frame(audit)
display(
    coherence.style.format(
        {
            "max abs error": "{:.3e}",
            "max rel error (%)": "{:.3f}",
            "max z-score": "{:.3f}",
        },
        na_rep="-",
    )
)'''


def normalize(path: Path) -> None:
    previous = nbf.read(path, as_version=4)
    model = path.parent.name
    product = _product_family(path)
    delta_crn = "delta_crn" in path.stem
    path_code = _path_code(previous)
    mode = "Price + Gradient" if delta_crn else "Price"
    title = (
        f"# {model.replace('_', ' ').title()} / "
        f"{product.replace('_', ' ').title()} {mode} Production Audit 01"
    )

    cells = [
        nbf.v4.new_markdown_cell(
            title + "\n\nValidate the first 100 production rows with the four independent engines."
        ),
        nbf.v4.new_markdown_cell(
            "## Load Data And Timings\n\n"
            "Load the production audit and display the common four-engine timing view."
        ),
        nbf.v4.new_code_cell(_setup_code(model, product, delta_crn)),
        nbf.v4.new_markdown_cell(
            "## Dataset Coherence\n\n"
            "Apply the production, native reproducibility, and statistical checks."
        ),
        nbf.v4.new_code_cell(COHERENCE_CODE),
    ]
    if path_code is not None:
        cells.extend((
            nbf.v4.new_markdown_cell(
                "## Path Reconstruction Check\n\n"
                "Reconstruct the exact Philox paths for one production row and reprice it."
            ),
            nbf.v4.new_code_cell(path_code),
        ))
    cells.append(nbf.v4.new_markdown_cell(_regeneration_markdown(previous)))

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["ai_factory"] = {
        "contract_version": 1,
        "model_family": model,
        "product_family": product,
        "delta_crn": delta_crn,
        "path_reconstruction": path_code is not None,
    }
    nbf.write(notebook, path)


def main() -> None:
    paths = sorted(NOTEBOOK_ROOT.glob("*/*.ipynb"))
    for path in paths:
        normalize(path)
    print(f"Normalized {len(paths)} validation notebooks.")


if __name__ == "__main__":
    main()
