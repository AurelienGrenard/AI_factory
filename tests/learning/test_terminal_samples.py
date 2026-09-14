"""Check terminal-sample conditioning, grouping and stream validation."""

import json
from pathlib import Path
import tempfile
import unittest

from learning.common.terminal_samples import iter_terminal_samples


def write_dataset(path: Path, *, bad_t: bool = False, written_count: int = 4) -> None:
    header = {
        "database_id": "samples_01",
        "model_family": "test_model",
        "row_count": 4,
        "time_convention": {"days_per_year": 252},
        "construction": {
            "parameter_count": 2,
            "paths_per_parameter": 2,
            "row_order": "parameter-major, then path-major",
        },
    }
    rows = [
        {
            "id": str(index + 1),
            "parameters": {"b": index // 2 + 1.0, "a": 0.5},
            "maturity_days": 252 if index % 2 == 0 else 126,
            "T": 5.0 if bad_t and index == 0 else (1.0 if index % 2 == 0 else 0.5),
            "values": {"state": float(index)},
        }
        for index in range(4)
    ]
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(header)[:-1] + ',\n"samples": [\n')
        for index, row in enumerate(rows[:written_count]):
            stream.write(json.dumps(row) + (",\n" if index < written_count - 1 else "\n"))
        stream.write("]\n}\n")


class TerminalSamplesTest(unittest.TestCase):
    def test_conditioning_and_group_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            write_dataset(path)
            pairs = list(iter_terminal_samples(path, seed=123))
        self.assertEqual(len(pairs), 4)
        schema = pairs[0][0]
        self.assertEqual(schema.conditioning_names, ("a", "b", "T"))
        self.assertEqual(pairs[0][1].conditioning, (0.5, 1.0, 1.0))
        self.assertEqual(pairs[1][1].conditioning, (0.5, 1.0, 0.5))
        self.assertEqual(pairs[0][1].parameter_group, pairs[1][1].parameter_group)
        self.assertEqual(pairs[0][1].split, pairs[1][1].split)
        self.assertEqual(pairs[2][1].split, pairs[3][1].split)

    def test_rejects_wrong_t(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            write_dataset(path, bad_t=True)
            with self.assertRaisesRegex(ValueError, "T differs"):
                list(iter_terminal_samples(path))

    def test_rejects_declared_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            write_dataset(path, written_count=3)
            with self.assertRaisesRegex(ValueError, "Dataset has 3 rows"):
                list(iter_terminal_samples(path))


if __name__ == "__main__":
    unittest.main()
