from __future__ import annotations

import json
from pathlib import Path

from tools.notebooks.build_validation_audits import COHERENCE_CODE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks" / "validation"
ENGINE_LABELS = ("cpp cuda", "pytorch gpu", "cpp cpu", "pytorch cpu")


def _notebooks() -> list[Path]:
    return sorted(NOTEBOOK_ROOT.glob("*/*.ipynb"))


def test_one_notebook_exists_for_every_production_result_mode() -> None:
    expected = 0
    for data_directory in (
        PROJECT_ROOT / "registry/production/results"
    ).glob("*/*/data"):
        expected += len({"delta_crn" in path.stem for path in data_directory.glob("*.json")})
    assert len(_notebooks()) == expected


def test_validation_notebooks_follow_the_canonical_spine() -> None:
    for path in _notebooks():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        metadata = notebook["metadata"]["ai_factory"]
        assert metadata["contract_version"] == 1
        assert metadata["model_family"] == path.parent.name
        cells = notebook["cells"]
        headings = [
            "".join(cell["source"]).splitlines()[0]
            for cell in cells if cell["cell_type"] == "markdown"
        ]
        expected = [
            headings[0],
            "## Load Data And Timings",
            "## Dataset Coherence",
        ]
        if metadata["path_reconstruction"]:
            expected.append("## Path Reconstruction Check")
        expected.append("## Regeneration Commands")
        assert headings == expected, (path, headings)
        assert "Economic Diagnostics" not in "\n".join(headings)
        assert "In/Out Parity" not in "\n".join(headings)
        for index, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                assert index > 0 and cells[index - 1]["cell_type"] == "markdown", path
                assert cell["execution_count"] is not None, path
                assert cell["outputs"], path
        code_cells = ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]
        assert code_cells[1] == COHERENCE_CODE
        assert "load_production_audit" in code_cells[0]
        assert all(
            output.get("output_type") != "error"
            for cell in cells for output in cell.get("outputs", [])
        )


def test_notebook_timing_order_is_stable() -> None:
    # The public frame is checked directly to avoid relying on notebook output HTML.
    from tools.validation.audit import ENGINE_ORDER

    assert tuple(label for _, label in ENGINE_ORDER) == ENGINE_LABELS
