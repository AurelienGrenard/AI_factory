from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tools.registry.model.slicing import write_model_slice
from tools.registry.product.slicing import write_product_slice


def _write_source(
    root: Path,
    *,
    kind: str,
    family: str,
    database_id: str,
    row_key: str,
) -> None:
    base = root / "registry" / "production" / kind / family
    data = base / "data" / f"{database_id}.json"
    specification = base / "specifications" / f"{database_id}.yaml"
    data.parent.mkdir(parents=True)
    specification.parent.mkdir(parents=True)
    payload = {
        "format": f"ai_factory.{kind}.v1",
        "database_id": database_id,
        f"{kind[:-1]}_family": family.replace("_", " ").title(),
        "row_count": 3,
        row_key: [{"id": f"{index:06d}", "parameters": {}} for index in range(1, 4)],
    }
    data.write_text(json.dumps(payload), encoding="utf-8")
    specification.write_text("format: test\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("kind", "family", "source_id", "target_id", "writer", "row_key"),
    [
        (
            "models",
            "heston",
            "heston_01",
            "heston_01__first_2",
            write_model_slice,
            "models",
        ),
        (
            "products",
            "american_puts",
            "american_puts_01",
            "american_puts_01__first_2",
            write_product_slice,
            "products",
        ),
    ],
)
def test_parameter_slice_is_exact_and_self_describing(
    tmp_path: Path,
    kind: str,
    family: str,
    source_id: str,
    target_id: str,
    writer,
    row_key: str,
) -> None:
    (tmp_path / "registry").mkdir()
    (tmp_path / "src_cpp").mkdir()
    _write_source(
        tmp_path,
        kind=kind,
        family=family,
        database_id=source_id,
        row_key=row_key,
    )

    target = writer(
        project_root=tmp_path,
        source_id=source_id,
        target_id=target_id,
        row_count=2,
    )
    data = json.loads(target.read_text(encoding="utf-8"))

    assert data["database_id"] == target_id
    assert data["row_count"] == 2
    assert [row["id"] for row in data[row_key]] == ["000001", "000002"]
    assert data["generation_script"].endswith(f"/{target_id}.py")


def test_slice_rejects_missing_rows(tmp_path: Path) -> None:
    (tmp_path / "registry").mkdir()
    (tmp_path / "src_cpp").mkdir()
    _write_source(
        tmp_path,
        kind="models",
        family="heston",
        database_id="heston_01",
        row_key="models",
    )

    with pytest.raises(ValueError, match="only has 3"):
        write_model_slice(
            project_root=tmp_path,
            source_id="heston_01",
            target_id="heston_01__first_4",
            row_count=4,
        )


def test_committed_validation_parameter_slices_are_exact_production_prefixes() -> None:
    project_root = Path(__file__).resolve().parents[2]
    for kind, row_key in (
        ("curves", "curves"),
        ("models", "models"),
        ("products", "products"),
    ):
        for json_path in (
            project_root / "registry" / "validation" / kind
        ).rglob("data/*.json"):
            yaml_path = (
                json_path.parent.parent / "specifications" / f"{json_path.stem}.yaml"
            )
            validation = json.loads(json_path.read_text(encoding="utf-8"))
            specification = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            production_path = project_root / specification["source_json_path"]
            production = json.loads(production_path.read_text(encoding="utf-8"))
            row_count = validation["row_count"]
            assert validation[row_key] == production[row_key][:row_count], json_path
