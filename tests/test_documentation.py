from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    "docs/architecture.md",
    "docs/production_dataset.md",
    "docs/validation_dataset.md",
    "docs/certification.md",
    "docs/code_guide.md",
)


def test_current_documentation_read_order_is_explicit() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    positions = [readme.index(path) for path in CURRENT_DOCS]
    assert positions == sorted(positions)
    assert "the journal is historical" in readme


def test_repository_agent_instructions_enforce_the_same_read_order() -> None:
    instructions = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "read `README.md`" in instructions
    positions = [instructions.index(f"`{path}`") for path in CURRENT_DOCS]
    assert positions == sorted(positions)
    assert "neither overrides the current contract" in " ".join(instructions.split())


def test_local_markdown_document_links_exist() -> None:
    for owner in [PROJECT_ROOT / "README.md", *(PROJECT_ROOT / "docs").glob("*.md")]:
        text = owner.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)]+\.md)(?:#[^)]+)?\)", text):
            resolved = (owner.parent / target).resolve()
            assert resolved.is_file(), (owner, target)


def test_current_docs_do_not_reference_superseded_notebook_contracts() -> None:
    current = "\n".join(
        (PROJECT_ROOT / path).read_text(encoding="utf-8")
        for path in CURRENT_DOCS
    )
    forbidden = (
        "exactly these six rows",
        "copied from the matching canonical Heston",
        "normalize_timing_cells.py",
        "build_autocall_audits.py",
        "build_barrier_audits.py",
        "build_rate_audits.py",
    )
    assert all(value not in current for value in forbidden)
    assert "tools/notebooks/build_validation_audits.py" in current
    assert "tools.validation.audit.coherence_frame" in current
