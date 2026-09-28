"""Generate terminal observations from a completed conditional GAN run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from learning.common.data.terminal_cache import TerminalTensorSchema
from learning.common.data.terminal_transforms import TerminalTransform

from .networks import build_generator
from .noise import sample_noise


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _conditions(path: Path) -> list[dict[str, float]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("conditions")
    if not isinstance(value, list) or not value:
        raise ValueError(
            "Condition input must be a non-empty list or "
            "{'conditions': [...]}"
        )
    if not all(isinstance(row, dict) for row in value):
        raise ValueError("Each condition must be a mapping")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("conditions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
    )
    parser.add_argument("--samples-per-condition", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    if arguments.samples_per_condition <= 0:
        parser.error("--samples-per-condition must be positive")
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be positive")

    checkpoint_path = arguments.run_directory / (
        "best_model.pt"
        if arguments.checkpoint == "best"
        else "last_model.pt"
    )
    device = _device(arguments.device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    schema = TerminalTensorSchema.from_dict(
        checkpoint["tensor_schema"]
    )
    transform = TerminalTransform.from_dict(
        checkpoint["transform"]
    )
    generator = build_generator(
        checkpoint["generator"],
        condition_dimension=len(schema.conditioning_names),
        observation_dimension=len(schema.observable_names),
    ).to(device)
    generator.load_state_dict(
        checkpoint["ema_generator_state_dict"]
    )
    generator.eval()

    configured = _conditions(arguments.conditions)
    expected = set(schema.conditioning_names)
    normalized_rows: list[list[float]] = []
    original_rows: list[dict[str, float]] = []
    for position, row in enumerate(configured):
        if set(row) != expected:
            raise ValueError(
                f"Condition {position} fields differ; "
                f"expected {schema.conditioning_names}"
            )
        ordered = [
            float(row[name]) for name in schema.conditioning_names
        ]
        if not all(math.isfinite(value) for value in ordered):
            raise ValueError(
                f"Condition {position} contains a non-finite value"
            )
        normalized_rows.extend(
            [ordered] * arguments.samples_per_condition
        )
        original_rows.extend(
            [
                {
                    name: float(row[name])
                    for name in schema.conditioning_names
                }
            ]
            * arguments.samples_per_condition
        )

    raw_conditions = torch.tensor(
        normalized_rows, dtype=torch.float32
    )
    random = torch.Generator(device="cpu")
    random.manual_seed(arguments.seed)
    generated_rows: list[list[float]] = []
    with torch.no_grad():
        for start in range(
            0, raw_conditions.shape[0], arguments.batch_size
        ):
            batch = raw_conditions[
                start : start + arguments.batch_size
            ].to(device)
            conditions = transform.normalize_conditions(batch)
            noise = sample_noise(
                batch.shape[0],
                generator.latent_dimension,
                generator.noise_distribution,
                random=random,
                device=device,
                uniform_minimum=generator.noise_minimum,
                uniform_maximum=generator.noise_maximum,
            )
            normalized = generator(conditions, noise)
            generated = transform.denormalize_observations(
                normalized
            )
            if not bool(torch.isfinite(generated).all()):
                raise FloatingPointError(
                    "Generator produced non-finite observations"
                )
            generated_rows.extend(
                generated.cpu().tolist()
            )

    output = {
        "schema_version": 1,
        "run_directory": str(
            arguments.run_directory.resolve()
        ),
        "checkpoint": arguments.checkpoint,
        "seed": arguments.seed,
        "conditioning_names": list(schema.conditioning_names),
        "observable_names": list(schema.observable_names),
        "samples": [
            {
                "condition": condition,
                "values": {
                    name: float(value)
                    for name, value in zip(
                        schema.observable_names,
                        generated,
                        strict=True,
                    )
                },
            }
            for condition, generated in zip(
                original_rows,
                generated_rows,
                strict=True,
            )
        ],
    }
    encoded = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = arguments.output.with_suffix(
            arguments.output.suffix + ".tmp"
        )
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(arguments.output)
        print(f"samples: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
