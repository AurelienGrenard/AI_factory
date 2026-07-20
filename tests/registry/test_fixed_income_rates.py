from __future__ import annotations
import json, math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RATE_MODELS=("hull_white","cir","cir_plus_plus","g2_plus_plus")
CURVE_MODELS={"hull_white","cir_plus_plus","g2_plus_plus"}
CAPLET_MODELS=("hull_white","cir","cir_plus_plus","g2_plus_plus")
def load(path:Path):return json.loads(path.read_text())
def result_data(model:str,product:str,needle:str):return load(next((ROOT/f"registry/validation/results/{model}/{product}/data").glob(f"*{needle}*.json")))
def max_abs(left,right):return max(abs(a["outputs"]["price"]-b["outputs"]["price"]) for a,b in zip(left,right))
def max_z(left,right):
    values=[]
    for a,b in zip(left,right):
        difference=abs(a["outputs"]["price"]-b["outputs"]["price"]);scale=math.hypot(a["outputs"]["standard_error"],b["outputs"]["standard_error"])
        if scale>0:values.append(difference/scale)
        elif difference>1e-12:values.append(float("inf"))
    return max(values,default=0.0)

def test_cir_grid_and_feller_ratio():
    rows=load(ROOT/"registry/production/models/cir/data/cir_01.json")["models"]
    assert len(rows)==1000
    for row in rows:
        p=row["parameters"];ratio=2*p["kappa"]*p["theta"]/p["volatility"]**2
        assert 0.1<=p["kappa"]<=2.0 and 0.005<=p["theta"]<=0.12 and 0.001<=p["initial_rate"]<=0.12
        assert 1/6-1e-12<=ratio<=10+1e-12

    shifted_rows=load(ROOT/"registry/production/models/cir_plus_plus/data/cir_plus_plus_01.json")["models"]
    assert len(shifted_rows)==1000
    for row in shifted_rows:
        p=row["parameters"];ratio=2*p["kappa"]*p["theta"]/p["volatility"]**2
        assert 0.1<=p["kappa"]<=2.0 and 0.005<=p["theta"]<=0.12 and 0.001<=p["initial_factor"]<=0.12
        assert 1/6-1e-12<=ratio<=10+1e-12

def test_g2_plus_plus_grid():
    rows=load(ROOT/"registry/production/models/g2_plus_plus/data/g2_plus_plus_01.json")["models"]
    assert len(rows)==1000
    for row in rows:
        p=row["parameters"]
        assert 0.02<=p["mean_reversion_x"]<=0.5
        assert 0.002<=p["volatility_x"]<=0.03
        assert 0.05<=p["mean_reversion_y"]<=1.0
        assert 0.002<=p["volatility_y"]<=0.03
        assert -0.95<=p["rho"]<=0.5

def test_black_76_grid_and_shifted_forwards_are_well_defined():
    models=load(ROOT/"registry/production/models/black_76/data/black_76_01.json")["models"]
    products=load(ROOT/"registry/production/products/caplets/data/caplets_01.json")["products"]
    assert len(models)==len(products)==1000
    for model,product in zip(models,products,strict=True):
        m=model["parameters"];p=product["parameters"]
        assert 0.05<=m["volatility"]<=0.80
        assert 0.03<=m["displacement"]<=0.06
        assert 0.25<=p["fixing_time"]<=10.0
        assert p["accrual_period"]==0.5 and 0.0<=p["strike"]<=0.08

def test_rate_product_grids_have_1000_rows():
    for product in ("zero_coupon_bonds","interest_rate_swaps","swaptions","bermudan_swaptions"):
        rows=load(ROOT/f"registry/production/products/{product}/data/{product}_01.json")["products"]
        assert len(rows)==1000
        if product == "zero_coupon_bonds":
            assert all(1/12 <= r["parameters"]["maturity"] <= 30 for r in rows)
            continue
        assert all(2<=r["parameters"]["payment_count"]<=20 for r in rows)
        if product == "bermudan_swaptions":
            assert all(
                2 <= row["parameters"]["exercise_count"] <= 8
                and row["parameters"]["payment_count"]
                    >= row["parameters"]["exercise_count"] + 2
                for row in rows
            )

def test_native_and_cross_engine_rate_coherence():
    for model in RATE_MODELS:
        for product in ("zero_coupon_bonds","interest_rate_swaps","swaptions","bermudan_swaptions"):
            gpu=result_data(model,product,"cpp_gpu")["results"];cpu=result_data(model,product,"cpp_cpu")["results"]
            pyg=result_data(model,product,"python_gpu")["results"];pyc=result_data(model,product,"python_cpu")["results"]
            assert max_abs(cpu,gpu)<1e-12
            assert max_z(pyc,pyg)<4.0
            assert max_z(gpu,pyg)<4.0
    for model in CAPLET_MODELS:
        gpu=result_data(model,"caplets","cpp_gpu")["results"];cpu=result_data(model,"caplets","cpp_cpu")["results"]
        pyg=result_data(model,"caplets","python_gpu")["results"];pyc=result_data(model,"caplets","python_cpu")["results"]
        assert max_abs(cpu,gpu)<1e-12
        assert max_z(pyc,pyg)<4.0
        assert max_z(gpu,pyg)<4.0
    for product in ("caplets","swaptions"):
        gpu=result_data("black_76",product,"cpp_gpu")["results"]
        for engine in ("cpp_cpu","python_gpu","python_cpu"):
            assert max_abs(gpu,result_data("black_76",product,engine)["results"])<1e-12

def test_production_heads_are_exactly_regenerated():
    for model in RATE_MODELS:
        for product in ("zero_coupon_bonds","interest_rate_swaps","swaptions","bermudan_swaptions"):
            production=load(next((ROOT/f"registry/production/results/{model}/{product}/data").glob("*.json")))["results"][:100]
            assert production==result_data(model,product,"cpp_gpu")["results"]
    for model in CAPLET_MODELS:
        production=load(next((ROOT/f"registry/production/results/{model}/caplets/data").glob("*.json")))["results"][:100]
        assert production==result_data(model,"caplets","cpp_gpu")["results"]
    for product in ("caplets","swaptions"):
        production=load(next((ROOT/f"registry/production/results/black_76/{product}/data").glob("*.json")))["results"][:100]
        assert production==result_data("black_76",product,"cpp_gpu")["results"]

def test_curve_links_only_where_required():
    for model in RATE_MODELS:
        for product in ("zero_coupon_bonds","interest_rate_swaps","swaptions","bermudan_swaptions"):
            row=result_data(model,product,"cpp_gpu")["results"][0]
            assert ("curve_id" in row)==(model in CURVE_MODELS)
    for model in CAPLET_MODELS:
        row=result_data(model,"caplets","cpp_gpu")["results"][0]
        assert ("curve_id" in row)==(model in CURVE_MODELS)
    assert "curve_id" in result_data("black_76","caplets","cpp_gpu")["results"][0]

def test_fixed_income_validation_timings_are_hot_benchmarks():
    repetitions = {
        "zero_coupon_bonds": 5,
        "interest_rate_swaps": 5,
        "swaptions": 5,
        "bermudan_swaptions": 3,
    }
    for model in RATE_MODELS:
        for product, expected in repetitions.items():
            for engine in ("cpp_gpu", "cpp_cpu", "python_gpu", "python_cpu"):
                timing = result_data(model, product, engine)["timing"]
                assert timing["wall_seconds"] > 0.0
                assert timing["benchmark_seconds"] > 0.0
                assert timing["benchmark_repetitions"] == expected
                if product in ("zero_coupon_bonds", "interest_rate_swaps"):
                    assert timing["benchmark_row_count"] == 100_000
                    assert timing["benchmark_workload"] == "duplicated validation slice"
    for model in CAPLET_MODELS:
        for engine in ("cpp_gpu", "cpp_cpu", "python_gpu", "python_cpu"):
            timing=result_data(model,"caplets",engine)["timing"]
            assert timing["benchmark_repetitions"]==5
            assert timing["benchmark_row_count"]==100
            assert timing["benchmark_workload"]=="validation slice"
    for product in ("caplets","swaptions"):
        for engine in ("cpp_gpu", "cpp_cpu", "python_gpu", "python_cpu"):
            timing=result_data("black_76",product,engine)["timing"]
            assert timing["benchmark_repetitions"]==5
            assert timing["benchmark_row_count"]==100_000
            assert timing["benchmark_workload"]=="duplicated validation slice"

def test_rate_notebooks_executed_without_errors():
    for model in RATE_MODELS:
        paths = [
            ROOT / f"notebooks/validation/{model}/2026-07-14_{model}_{product}_production_audit_01.ipynb"
            for product in ("zero_coupon_bonds", "interest_rate_swaps", "swaptions", "bermudan_swaptions")
        ]
        assert all(path.is_file() for path in paths)
        for path in paths:
            notebook=load(path)
            assert all(output.get("output_type")!="error" for cell in notebook["cells"] for output in cell.get("outputs",[]))
    extra_paths=[
        ROOT/f"notebooks/validation/{model}/2026-07-16_{model}_caplets_production_audit_01.ipynb"
        for model in CAPLET_MODELS
    ]+[
        ROOT/f"notebooks/validation/black_76/2026-07-16_black_76_{product}_production_audit_01.ipynb"
        for product in ("caplets","swaptions")
    ]
    for path in extra_paths:
        notebook=load(path)
        assert all(
            output.get("output_type")!="error"
            for cell in notebook["cells"]
            for output in cell.get("outputs",[])
        )
