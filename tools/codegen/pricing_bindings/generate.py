#!/usr/bin/env python3
"""Generate prototype pricing bindings into a caller-owned output folder."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from manifest import BINDINGS, Binding


SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "templates"


def time_fields(binding: Binding) -> dict[str, str]:
    if binding.time_kind == "exact":
        return {
            "time_parameter_declarations": "    float day_fraction,\n",
            "time_configuration": (
                "    const simulation::ExactTransitionTimeConfiguration "
                "time_configuration{\n"
                "        day_fraction,\n"
                "    };\n"
            ),
            "instantiation_time_signature": (
                "    float, unsigned int, std::size_t, std::uint64_t,"
            ),
        }
    if binding.time_kind == "fixed":
        return {
            "time_parameter_declarations": (
                "    float dt,\n"
                "    std::uint32_t simulation_steps_per_day,\n"
            ),
            "time_configuration": (
                "    const simulation::FixedStepTimeConfiguration "
                "time_configuration{\n"
                "        dt,\n"
                "        simulation_steps_per_day,\n"
                "    };\n"
            ),
            "instantiation_time_signature": (
                "    float, std::uint32_t, unsigned int, std::size_t, "
                "std::uint64_t,"
            ),
        }
    raise ValueError(f"Unsupported time kind: {binding.time_kind}")


def render(template: str, binding: Binding) -> str:
    product_words = binding.product.replace("_", " ")
    values = {
        **binding.__dict__,
        **time_fields(binding),
        "product_comment": binding.product.replace("_", "-"),
        "header_product_comment": binding.product.replace(
            "_", "-"
        ).capitalize(),
        "operation_product": product_words,
    }
    return template.format(**values)


def generate(output_root: Path) -> list[Path]:
    header_template = (TEMPLATE_DIR / "header.tpl").read_text()
    source_template = (TEMPLATE_DIR / "source.tpl").read_text()
    generated: list[Path] = []
    for binding in BINDINGS:
        if not binding.sided:
            raise ValueError("The prototype currently supports sided bindings only.")
        destination = output_root / "src" / "model" / "equity" / binding.model
        destination.mkdir(parents=True, exist_ok=True)
        header = destination / f"{binding.product}.cuh"
        source = destination / f"{binding.product}.cu"
        header.write_text(render(header_template, binding))
        source.write_text(render(source_template, binding))
        generated.extend((header, source))
    return generated


def compare(generated: list[Path], output_root: Path, reference_root: Path) -> int:
    mismatch_count = 0
    for generated_path in generated:
        relative_path = generated_path.relative_to(output_root)
        reference_path = reference_root / relative_path
        reference = reference_path.read_text()
        current = generated_path.read_text()
        if current == reference:
            continue
        mismatch_count += 1
        print(f"Mismatch: {relative_path}")
        print("".join(difflib.unified_diff(
            reference.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile=str(reference_path),
            tofile=str(generated_path),
        )))
    return mismatch_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ai_factory-pricing-bindings"),
    )
    parser.add_argument("--compare-root", type=Path)
    arguments = parser.parse_args()

    generated = generate(arguments.output)
    for path in generated:
        print(path)
    if arguments.compare_root is None:
        return 0
    return 1 if compare(
        generated,
        arguments.output,
        arguments.compare_root,
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
