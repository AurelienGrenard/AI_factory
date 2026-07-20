from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_registry_databases_have_complete_triplets() -> None:
    for tier in ("production", "validation"):
        for kind in ("models", "curves", "products", "results"):
            root = PROJECT_ROOT / "registry" / tier / kind
            ids = {
                "data": {path.stem for path in root.rglob("*.json")},
                "specifications": {path.stem for path in root.rglob("*.yaml")},
                "generators": {path.stem for path in root.rglob("*.py")},
            }
            assert ids["data"] == ids["specifications"] == ids["generators"]


def test_validation_does_not_store_production_result_slices() -> None:
    root = PROJECT_ROOT / "registry" / "validation" / "results"
    trailing_slices = [
        path for path in root.rglob("*")
        if path.is_file() and path.stem.endswith("__first_100")
    ]
    assert trailing_slices == []


def test_every_production_pair_has_all_source_backends() -> None:
    root = PROJECT_ROOT / "registry" / "production" / "results"
    for result_path in root.rglob("*.json"):
        relative = result_path.relative_to(root)
        model, product = relative.parts[:2]
        expected = (
            PROJECT_ROOT / f"src_cpp/ai_factory/cpu/{model}/{product}.hpp",
            PROJECT_ROOT / f"src_cpp/ai_factory/cpu/{model}/{product}.cpp",
            PROJECT_ROOT / f"src_cpp/ai_factory/cuda/{model}/{product}.cu",
            PROJECT_ROOT / f"src_python/ai_factory/pytorch/{model}/{product}.py",
            PROJECT_ROOT / f"tools/registry/result/{model}/{product}.py",
        )
        assert all(path.is_file() for path in expected), result_path


def test_cmake_lists_every_native_implementation_unit() -> None:
    source_root = PROJECT_ROOT / "src_cpp"
    cmake = (source_root / "CMakeLists.txt").read_text(encoding="utf-8")
    missing = [
        path.relative_to(source_root).as_posix()
        for path in (source_root / "ai_factory").rglob("*")
        if path.suffix in {".cpp", ".cu"}
        and path.relative_to(source_root).as_posix() not in cmake
    ]
    assert missing == []


def test_result_json_generation_scripts_exist() -> None:
    for path in (PROJECT_ROOT / "registry").rglob("results/**/*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        generator = PROJECT_ROOT / data["generation_script"]
        assert generator.is_file(), path


def test_validation_notebooks_are_grouped_by_model() -> None:
    root = PROJECT_ROOT / "notebooks" / "validation"
    assert list(root.glob("*.ipynb")) == []
    for path in root.glob("*/*.ipynb"):
        assert f"_{path.parent.name}_" in path.name, path


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_source_python_does_not_depend_on_tools_or_registry() -> None:
    for path in (PROJECT_ROOT / "src_python").rglob("*.py"):
        forbidden = {
            module for module in _imported_modules(path)
            if module == "tools" or module.startswith("tools.")
            or module == "registry" or module.startswith("registry.")
        }
        assert forbidden == set(), (path, forbidden)


def test_result_modules_do_not_import_other_model_families() -> None:
    root = PROJECT_ROOT / "tools/registry/result"
    model_families = {
        path.name for path in root.iterdir()
        if path.is_dir() and path.name != "common"
    }
    for family in model_families:
        for path in (root / family).glob("*.py"):
            imported_families = {
                module.split(".")[3]
                for module in _imported_modules(path)
                if module.startswith("tools.registry.result.")
                and len(module.split(".")) > 3
            }
            assert imported_families <= {"common", family}, (
                path, imported_families - {"common", family}
            )


def test_common_result_tools_do_not_import_specialized_results() -> None:
    root = PROJECT_ROOT / "tools/registry/result/common"
    for path in root.glob("*.py"):
        specialized = {
            module for module in _imported_modules(path)
            if module.startswith("tools.registry.result.")
            and not module.startswith("tools.registry.result.common")
        }
        assert specialized == set(), (path, specialized)
