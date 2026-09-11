"""Run selected native paired generators on two staged input rows, never publish."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.datasets.generate_catalog import check_outputs, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--target", action="append", help="restrict the bounded recipe selection")
    parser.add_argument("--original-calendar", action="store_true", help="diagnostic only: retain input maturities")
    args = parser.parse_args()
    targets = {
        "generate_bates_athena_autocalls_01_price_delta",
        "generate_black_scholes_european_calls_01_price_delta",
        "generate_black_scholes_geometric_asian_calls_01_price_delta",
        "generate_heston_european_calls_01_price_delta",
        "generate_heston_american_puts_01_price_delta",
        "generate_sabr_athena_autocalls_01_price_delta",
        "generate_variance_gamma_european_calls_01_price_delta",
    }
    for job in inventory(ROOT, {"price_delta"}, set(), set(args.target) if args.target else targets):
        with tempfile.TemporaryDirectory(prefix="ai_factory_delta_generator_") as temporary:
            work = Path(temporary)
            for name in job["inputs"]:
                document = json.loads((ROOT / name).read_text())
                role = "models" if "models" in document else "products"
                document[role] = document[role][:2]
                document["row_count"] = 2
                if role == "products" and not args.original_calendar:
                    for row in document[role]:
                        parameters = row["parameters"]
                        parameters["maturity"] = 21
                        if "exercise_interval" in parameters:
                            parameters["maturity"] = 8
                            parameters["exercise_interval"] = 7
                        if "observation_interval" in parameters:
                            parameters["observation_interval"] = 7
                destination = work / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(document))
            subprocess.run([str(args.build.resolve() / job["target"])], cwd=work, check=True, timeout=90)
            closed = job["declared_method"]["engine"] == "equity_closed_form"
            job.update(rows=2, launch_plan={"paths_per_price": 0 if closed else 1048576},
                       previous={job["dataset"]: None, job["catalog"]: None})
            check_outputs(work, job)
            print(job["target"] + ": two rows, native outputs and frozen contract checked", flush=True)


if __name__ == "__main__":
    main()
