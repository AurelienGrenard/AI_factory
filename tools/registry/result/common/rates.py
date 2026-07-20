"""Shared registry orchestration for fixed-income swaps and swaptions."""
from __future__ import annotations
import ctypes, importlib, json, sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import yaml
from tools.common.native_library import load_cpp_library
from tools.registry.common.paths import registry_database_path, registry_relative_path
from tools.registry.common.timing import benchmark_pricing_call
from tools.registry.common.schema import (
    aligned_result_construction,
    analytic_time_grid,
    canonical_timing,
    database_reference,
    exact_transition_time_grid,
    primary_source_files,
)

PROJECT_ROOT=Path(__file__).resolve().parents[4]
SRC_PYTHON=PROJECT_ROOT/"src_python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0,str(SRC_PYTHON))
DEFAULT_NUM_PATHS=16_384
DEFAULT_FIRST_SEED=950_100_001
AUDIT_ROW_COUNT=100
PRODUCTION_ROW_COUNT=1_000
TARGET_DT=1.0/52.0
ANALYTIC_THROUGHPUT_ROW_COUNT=100_000
ANALYTIC_THROUGHPUT_REPETITIONS=5

class COutput(ctypes.Structure): _fields_=[("price",ctypes.c_double),("standard_error",ctypes.c_double)]
class CSwapTerms(ctypes.Structure): _fields_=[("start_time",ctypes.c_double),("accrual_period",ctypes.c_double),("fixed_rate",ctypes.c_double),("notional",ctypes.c_double),("payment_count",ctypes.c_int),("direction",ctypes.c_int)]
class CZeroCouponBondTerms(ctypes.Structure): _fields_=[("maturity",ctypes.c_double),("notional",ctypes.c_double)]
class CSwaptionTerms(ctypes.Structure): _fields_=[("expiry",ctypes.c_double),("accrual_period",ctypes.c_double),("fixed_rate",ctypes.c_double),("notional",ctypes.c_double),("payment_count",ctypes.c_int),("direction",ctypes.c_int)]
class CBermudanSwaptionTerms(ctypes.Structure): _fields_=[("first_exercise",ctypes.c_double),("exercise_period",ctypes.c_double),("accrual_period",ctypes.c_double),("fixed_rate",ctypes.c_double),("notional",ctypes.c_double),("exercise_count",ctypes.c_int),("payment_count",ctypes.c_int),("direction",ctypes.c_int)]
class CCapletTerms(ctypes.Structure): _fields_=[("fixing_time",ctypes.c_double),("accrual_period",ctypes.c_double),("strike",ctypes.c_double),("notional",ctypes.c_double)]
class CBlack76Model(ctypes.Structure): _fields_=[("volatility",ctypes.c_double),("displacement",ctypes.c_double)]
class CBlack76Curve(ctypes.Structure): _fields_=[("beta0",ctypes.c_double),("beta1",ctypes.c_double),("beta2",ctypes.c_double),("tau",ctypes.c_double)]
class CBlack76CapletRow(ctypes.Structure): _fields_=[("model",CBlack76Model),("curve",CBlack76Curve),("product",CCapletTerms)]
class CBlack76SwaptionRow(ctypes.Structure): _fields_=[("model",CBlack76Model),("curve",CBlack76Curve),("product",CSwaptionTerms)]
class CHullWhiteSwapRow(ctypes.Structure): _fields_=[("mean_reversion",ctypes.c_double),("volatility",ctypes.c_double),("beta0",ctypes.c_double),("beta1",ctypes.c_double),("beta2",ctypes.c_double),("tau",ctypes.c_double),("product",CSwapTerms)]
class CHullWhiteZeroCouponBondRow(ctypes.Structure): _fields_=[("mean_reversion",ctypes.c_double),("volatility",ctypes.c_double),("beta0",ctypes.c_double),("beta1",ctypes.c_double),("beta2",ctypes.c_double),("tau",ctypes.c_double),("product",CZeroCouponBondTerms)]
class CHullWhiteSwaptionRow(ctypes.Structure): _fields_=[("mean_reversion",ctypes.c_double),("volatility",ctypes.c_double),("beta0",ctypes.c_double),("beta1",ctypes.c_double),("beta2",ctypes.c_double),("tau",ctypes.c_double),("product",CSwaptionTerms),("seed",ctypes.c_uint64)]
class CHullWhiteBermudanSwaptionRow(ctypes.Structure): _fields_=[("mean_reversion",ctypes.c_double),("volatility",ctypes.c_double),("beta0",ctypes.c_double),("beta1",ctypes.c_double),("beta2",ctypes.c_double),("tau",ctypes.c_double),("product",CBermudanSwaptionTerms),("seed",ctypes.c_uint64)]
class CHullWhiteCapletRow(ctypes.Structure): _fields_=[("mean_reversion",ctypes.c_double),("volatility",ctypes.c_double),("beta0",ctypes.c_double),("beta1",ctypes.c_double),("beta2",ctypes.c_double),("tau",ctypes.c_double),("product",CCapletTerms),("seed",ctypes.c_uint64)]
class CCirModel(ctypes.Structure): _fields_=[("initial_rate",ctypes.c_double),("kappa",ctypes.c_double),("theta",ctypes.c_double),("volatility",ctypes.c_double)]
class CCirSwapRow(ctypes.Structure): _fields_=[("model",CCirModel),("product",CSwapTerms)]
class CCirZeroCouponBondRow(ctypes.Structure): _fields_=[("model",CCirModel),("product",CZeroCouponBondTerms)]
class CCirSwaptionRow(ctypes.Structure): _fields_=[("model",CCirModel),("product",CSwaptionTerms),("seed",ctypes.c_uint64)]
class CCirBermudanSwaptionRow(ctypes.Structure): _fields_=[("model",CCirModel),("product",CBermudanSwaptionTerms),("seed",ctypes.c_uint64)]
class CCirCapletRow(ctypes.Structure): _fields_=[("model",CCirModel),("product",CCapletTerms),("seed",ctypes.c_uint64)]
class CCirPlusPlusModel(ctypes.Structure): _fields_=[("initial_factor",ctypes.c_double),("kappa",ctypes.c_double),("theta",ctypes.c_double),("volatility",ctypes.c_double)]
class CShiftCurve(ctypes.Structure): _fields_=[("beta0",ctypes.c_double),("beta1",ctypes.c_double),("beta2",ctypes.c_double),("tau",ctypes.c_double)]
class CCirPlusPlusSwapRow(ctypes.Structure): _fields_=[("model",CCirPlusPlusModel),("curve",CShiftCurve),("product",CSwapTerms)]
class CCirPlusPlusZeroCouponBondRow(ctypes.Structure): _fields_=[("model",CCirPlusPlusModel),("curve",CShiftCurve),("product",CZeroCouponBondTerms)]
class CCirPlusPlusSwaptionRow(ctypes.Structure): _fields_=[("model",CCirPlusPlusModel),("curve",CShiftCurve),("product",CSwaptionTerms),("seed",ctypes.c_uint64)]
class CCirPlusPlusBermudanSwaptionRow(ctypes.Structure): _fields_=[("model",CCirPlusPlusModel),("curve",CShiftCurve),("product",CBermudanSwaptionTerms),("seed",ctypes.c_uint64)]
class CCirPlusPlusCapletRow(ctypes.Structure): _fields_=[("model",CCirPlusPlusModel),("curve",CShiftCurve),("product",CCapletTerms),("seed",ctypes.c_uint64)]
class CG2PlusPlusModel(ctypes.Structure): _fields_=[("mean_reversion_x",ctypes.c_double),("volatility_x",ctypes.c_double),("mean_reversion_y",ctypes.c_double),("volatility_y",ctypes.c_double),("rho",ctypes.c_double)]
class CG2PlusPlusSwapRow(ctypes.Structure): _fields_=[("model",CG2PlusPlusModel),("curve",CShiftCurve),("product",CSwapTerms)]
class CG2PlusPlusZeroCouponBondRow(ctypes.Structure): _fields_=[("model",CG2PlusPlusModel),("curve",CShiftCurve),("product",CZeroCouponBondTerms)]
class CG2PlusPlusSwaptionRow(ctypes.Structure): _fields_=[("model",CG2PlusPlusModel),("curve",CShiftCurve),("product",CSwaptionTerms),("seed",ctypes.c_uint64)]
class CG2PlusPlusBermudanSwaptionRow(ctypes.Structure): _fields_=[("model",CG2PlusPlusModel),("curve",CShiftCurve),("product",CBermudanSwaptionTerms),("seed",ctypes.c_uint64)]
class CG2PlusPlusCapletRow(ctypes.Structure): _fields_=[("model",CG2PlusPlusModel),("curve",CShiftCurve),("product",CCapletTerms),("seed",ctypes.c_uint64)]

@dataclass(frozen=True)
class RateConfig:
    model_family:str
    product_family:str
    uses_curve:bool
    stochastic:bool

def _path(tier,kind,section,database_id,suffix):return registry_database_path(PROJECT_ROOT,tier,kind,section,database_id,suffix)
def _read(path):return json.loads(path.read_text())
def _rows(data,key):return {r["id"]:r["parameters"] for r in data[key]}
def aligned_rows(count,uses_curve):
    return [{"id":f"{i+1:06d}","model_id":f"{i+1:06d}",**({"curve_id":f"{i+1:06d}"} if uses_curve else {}),"product_id":f"{i+1:06d}","seed":DEFAULT_FIRST_SEED+i} for i in range(count)]

def load_inputs(tier,model_id,product_id,row_count,curve_id=None):
    models=_rows(_read(_path(tier,"models","data",model_id,"json")),"models");products=_rows(_read(_path(tier,"products","data",product_id,"json")),"products");curves=_rows(_read(_path(tier,"curves","data",curve_id,"json")),"curves") if curve_id else {}
    return aligned_rows(row_count,curve_id is not None),models,curves,products

def _swap_terms(p):return CSwapTerms(p["start_time"],p["accrual_period"],p["fixed_rate"],p["notional"],p["payment_count"],p["direction"])
def _zero_coupon_terms(p):return CZeroCouponBondTerms(p["maturity"],p["notional"])
def _swaption_terms(p):return CSwaptionTerms(p["expiry"],p["accrual_period"],p["fixed_rate"],p["notional"],p["payment_count"],p["direction"])
def _bermudan_terms(p):return CBermudanSwaptionTerms(p["first_exercise"],p["exercise_period"],p["accrual_period"],p["fixed_rate"],p["notional"],p["exercise_count"],p["payment_count"],p["direction"])
def _caplet_terms(p):return CCapletTerms(p["fixing_time"],p["accrual_period"],p["strike"],p["notional"])
def _c_row(config,row,models,curves,products):
    m=models[row["model_id"]];p=products[row["product_id"]]
    if config.model_family == "black_76":
        c=curves[row["curve_id"]];model=CBlack76Model(m["volatility"],m["displacement"]);curve=CBlack76Curve(c["beta0"],c["beta1"],c["beta2"],c["tau"])
        return CBlack76CapletRow(model,curve,_caplet_terms(p)) if config.product_family=="caplets" else CBlack76SwaptionRow(model,curve,_swaption_terms(p))
    if config.model_family=="hull_white":
        c=curves[row["curve_id"]];head=(m["mean_reversion"],m["volatility"],c["beta0"],c["beta1"],c["beta2"],c["tau"])
        if config.product_family == "zero_coupon_bonds":
            return CHullWhiteZeroCouponBondRow(*head,_zero_coupon_terms(p))
        if config.product_family == "bermudan_swaptions":
            return CHullWhiteBermudanSwaptionRow(*head,_bermudan_terms(p),row["seed"])
        if config.product_family == "caplets":
            return CHullWhiteCapletRow(*head,_caplet_terms(p),row["seed"])
        return CHullWhiteSwaptionRow(*head,_swaption_terms(p),row["seed"]) if config.stochastic else CHullWhiteSwapRow(*head,_swap_terms(p))
    if config.model_family == "cir_plus_plus":
        cm=CCirPlusPlusModel(m["initial_factor"],m["kappa"],m["theta"],m["volatility"]);c=curves[row["curve_id"]];curve=CShiftCurve(c["beta0"],c["beta1"],c["beta2"],c["tau"])
        if config.product_family=="zero_coupon_bonds":return CCirPlusPlusZeroCouponBondRow(cm,curve,_zero_coupon_terms(p))
        if config.product_family=="bermudan_swaptions":return CCirPlusPlusBermudanSwaptionRow(cm,curve,_bermudan_terms(p),row["seed"])
        if config.product_family=="caplets":return CCirPlusPlusCapletRow(cm,curve,_caplet_terms(p),row["seed"])
        return CCirPlusPlusSwaptionRow(cm,curve,_swaption_terms(p),row["seed"]) if config.stochastic else CCirPlusPlusSwapRow(cm,curve,_swap_terms(p))
    if config.model_family == "g2_plus_plus":
        gm=CG2PlusPlusModel(m["mean_reversion_x"],m["volatility_x"],m["mean_reversion_y"],m["volatility_y"],m["rho"]);c=curves[row["curve_id"]];curve=CShiftCurve(c["beta0"],c["beta1"],c["beta2"],c["tau"])
        if config.product_family=="zero_coupon_bonds":return CG2PlusPlusZeroCouponBondRow(gm,curve,_zero_coupon_terms(p))
        if config.product_family=="bermudan_swaptions":return CG2PlusPlusBermudanSwaptionRow(gm,curve,_bermudan_terms(p),row["seed"])
        if config.product_family=="caplets":return CG2PlusPlusCapletRow(gm,curve,_caplet_terms(p),row["seed"])
        return CG2PlusPlusSwaptionRow(gm,curve,_swaption_terms(p),row["seed"]) if config.stochastic else CG2PlusPlusSwapRow(gm,curve,_swap_terms(p))
    cm=CCirModel(m["initial_rate"],m["kappa"],m["theta"],m["volatility"])
    if config.product_family == "zero_coupon_bonds":
        return CCirZeroCouponBondRow(cm,_zero_coupon_terms(p))
    if config.product_family == "bermudan_swaptions":
        return CCirBermudanSwaptionRow(cm,_bermudan_terms(p),row["seed"])
    if config.product_family == "caplets":
        return CCirCapletRow(cm,_caplet_terms(p),row["seed"])
    return CCirSwaptionRow(cm,_swaption_terms(p),row["seed"]) if config.stochastic else CCirSwapRow(cm,_swap_terms(p))

def cpp_outputs(config,rows,models,curves,products,*,num_paths,use_gpu):
    library=load_cpp_library();device="gpu" if use_gpu else "cpu"
    product_symbol={"zero_coupon_bonds":"zero_coupon_bond","interest_rate_swaps":"interest_rate_swap","swaptions":"swaption","bermudan_swaptions":"bermudan_swaption","caplets":"caplet"}[config.product_family]
    symbol=f"ai_factory_price_{config.model_family}_{product_symbol}_{device}_batch";function=getattr(library,symbol)
    if config.model_family == "black_76":
        row_class=CBlack76CapletRow if config.product_family=="caplets" else CBlack76SwaptionRow
    elif config.product_family == "zero_coupon_bonds":
        row_class={"hull_white":CHullWhiteZeroCouponBondRow,"cir":CCirZeroCouponBondRow,"cir_plus_plus":CCirPlusPlusZeroCouponBondRow,"g2_plus_plus":CG2PlusPlusZeroCouponBondRow}[config.model_family]
    elif config.product_family == "bermudan_swaptions":
        row_class={"hull_white":CHullWhiteBermudanSwaptionRow,"cir":CCirBermudanSwaptionRow,"cir_plus_plus":CCirPlusPlusBermudanSwaptionRow,"g2_plus_plus":CG2PlusPlusBermudanSwaptionRow}[config.model_family]
    elif config.product_family == "caplets":
        row_class={"hull_white":CHullWhiteCapletRow,"cir":CCirCapletRow,"cir_plus_plus":CCirPlusPlusCapletRow,"g2_plus_plus":CG2PlusPlusCapletRow}[config.model_family]
    else:
        row_class={('hull_white',False):CHullWhiteSwapRow,('hull_white',True):CHullWhiteSwaptionRow,('cir',False):CCirSwapRow,('cir',True):CCirSwaptionRow,('cir_plus_plus',False):CCirPlusPlusSwapRow,('cir_plus_plus',True):CCirPlusPlusSwaptionRow,('g2_plus_plus',False):CG2PlusPlusSwapRow,('g2_plus_plus',True):CG2PlusPlusSwaptionRow}[(config.model_family,config.stochastic)]
    args=[ctypes.POINTER(row_class),ctypes.c_size_t]
    if config.stochastic:args.append(ctypes.c_size_t)
    if config.model_family in {"cir","cir_plus_plus"} and config.stochastic:args.append(ctypes.c_double)
    args.append(ctypes.POINTER(COutput))
    if use_gpu:args.append(ctypes.POINTER(ctypes.c_double))
    function.argtypes=args;function.restype=ctypes.c_int
    if use_gpu:
        library.ai_factory_cuda_warmup.argtypes=[];library.ai_factory_cuda_warmup.restype=ctypes.c_int
        if library.ai_factory_cuda_warmup()!=0:raise RuntimeError("CUDA warm-up failed")
    row_array=(row_class*len(rows))(*[_c_row(config,r,models,curves,products) for r in rows]);outputs=(COutput*len(rows))();kernel=ctypes.c_double();call=[row_array,len(rows)]
    if config.stochastic:call.append(num_paths)
    if config.model_family in {"cir","cir_plus_plus"} and config.stochastic:call.append(TARGET_DT)
    call.append(outputs)
    if use_gpu:call.append(ctypes.byref(kernel))
    started=perf_counter();status=function(*call);wall=perf_counter()-started
    if status:raise RuntimeError("Native fixed-income pricing failed")
    timing={"wall_seconds":wall};
    if use_gpu:timing["kernel_seconds"]=kernel.value
    return [{"price":o.price,"standard_error":o.standard_error} for o in outputs],timing

def pytorch_outputs(config,rows,models,curves,products,*,num_paths,device):
    module=importlib.import_module(f"ai_factory.pytorch.{config.model_family}.{config.product_family}")
    kwargs={"num_paths":num_paths,"device":device}
    if config.model_family in {"cir","cir_plus_plus"} and config.stochastic:kwargs["target_dt"]=TARGET_DT
    return module.price_batch(rows,models,*(([curves,products]) if config.uses_curve else ([products])),**kwargs)

def generate_result(config:RateConfig,*,tier,model_id,product_id,result_version,engine,row_count,curve_id=None,num_paths=DEFAULT_NUM_PATHS,source_production_result=None):
    rows,models,curves,products=load_inputs(tier,model_id,product_id,row_count,curve_id)
    def price(selected_rows=rows):
        if engine.startswith("python_"):
            return pytorch_outputs(
                config, selected_rows, models, curves, products,
                num_paths=num_paths,
                device="cuda" if "_gpu_" in engine else "cpu",
            )
        return cpp_outputs(
            config, selected_rows, models, curves, products,
            num_paths=num_paths, use_gpu="_gpu_" in engine,
        )

    if tier == "validation":
        if not config.stochastic:
            outputs, timing = price()
            repeats = (
                ANALYTIC_THROUGHPUT_ROW_COUNT + len(rows) - 1
            ) // len(rows)
            throughput_rows = (rows * repeats)[:ANALYTIC_THROUGHPUT_ROW_COUNT]
            _, benchmark = benchmark_pricing_call(
                lambda: price(throughput_rows),
                repetitions=ANALYTIC_THROUGHPUT_REPETITIONS,
                warmup_runs=1,
            )
            timing.update({
                key: value
                for key, value in benchmark.items()
                if key.startswith("benchmark_") or key == "warmup_calls"
            })
            timing["benchmark_row_count"] = len(throughput_rows)
            timing["benchmark_workload"] = "duplicated validation slice"
        else:
            repetitions = (
                3 if config.product_family == "bermudan_swaptions" else 5
            )
            outputs, timing = benchmark_pricing_call(
                price,
                repetitions=repetitions,
                warmup_runs=1,
            )
            timing["benchmark_row_count"] = len(rows)
            timing["benchmark_workload"] = "validation slice"
    else:
        if engine.startswith("cpp_") and "_gpu_" in engine and rows:
            cpp_outputs(
                config,
                rows[:min(AUDIT_ROW_COUNT, len(rows))],
                models,
                curves,
                products,
                num_paths=num_paths,
                use_gpu=True,
            )
        outputs, timing = price()
    timing=canonical_timing(timing)
    ids=[model_id]+([curve_id] if curve_id else [])+[product_id,f"{engine}_{result_version}"];database_id="__".join(ids);json_path=_path(tier,"results","data",database_id,"json");yaml_path=_path(tier,"results","specifications",database_id,"yaml");generator=registry_relative_path(PROJECT_ROOT,tier,"results","generators",database_id,"py");json_path.parent.mkdir(parents=True,exist_ok=True);yaml_path.parent.mkdir(parents=True,exist_ok=True)
    result_rows=[{**r,"outputs":o} for r,o in zip(rows,outputs,strict=True)]
    if "analytic" in engine:
        result_rows=[{key:value for key,value in row.items() if key!="seed"} for row in result_rows]
    construction=aligned_result_construction(DEFAULT_FIRST_SEED if config.stochastic else None)
    construction["purpose"] = (
        "initial_curve_reproduction" if config.product_family == "zero_coupon_bonds"
        else "curve_implied_pricing" if config.product_family == "interest_rate_swaps"
        else "model_dependent_pricing"
    )
    model_database=database_reference(PROJECT_ROOT,tier,"models",model_id)
    product_database=database_reference(PROJECT_ROOT,tier,"products",product_id)
    curve_database=database_reference(PROJECT_ROOT,tier,"curves",curve_id) if curve_id else None
    base={"format":"ai_factory.results.v1","database_id":database_id,"status":"priced","specification":yaml_path.relative_to(PROJECT_ROOT).as_posix(),"generation_script":generator,"row_count":len(rows),"model_database":model_database,**({"curve_database":curve_database} if curve_database else {}),"product_database":product_database,"result_construction":construction,"engine":engine,"timing":timing,"results":result_rows};json_path.write_text(json.dumps(base,indent=2)+"\n")
    source=primary_source_files(model_id,product_id,engine)
    summary={"row_count":len(rows),"model":config.model_family.replace('_',' ').title(),"payoff":config.product_family.replace('_',' ').title(),"engine":engine,"device":"gpu" if "_gpu_" in engine else "cpu","source_files":source};
    if curve_id:summary["curve"]="Nelson Siegel"
    if config.stochastic:summary["num_paths"]=num_paths
    if source_production_result:summary["source_production_result"]=source_production_result
    if config.product_family == "bermudan_swaptions":
        summary["references"]=[{
            "topic":"Longstaff-Schwartz exercise policy",
            "reference":{
                "authors":"Longstaff, F. A.; Schwartz, E. S.",
                "year":2001,
                "title":"Valuing American Options by Simulation: A Simple Least-Squares Approach",
            },
        }]
        if engine.startswith("cpp_"):
            summary["references"].append({
                "topic":"C++ Philox counter-based random numbers",
                "reference":{
                    "authors":"Salmon, J. K.; Moraes, M. A.; Dror, R. O.; Shaw, D. E.",
                    "year":2011,
                    "title":"Parallel Random Numbers: As Easy as 1, 2, 3",
                },
            })
    exercise = None
    if config.product_family == "bermudan_swaptions":
        exercise={
            "method":"Longstaff-Schwartz",
            "dates":"first_exercise + j * exercise_period",
            "regression_target":"Realized future exercise cashflow discounted to the current exercise date",
            "basis":"Laguerre polynomials of degree 0 to 3 on par swap rate / 0.04",
            "regression_sample":"In-the-money paths only",
        }
    standard_error = (
        "Monte Carlo standard error"
        if config.stochastic
        else "Exactly zero for deterministic analytic pricing"
    )
    event_label = "fixing dates" if config.product_family == "caplets" else "exercise dates"
    spec={"title":database_id,"format":"ai_factory.results.v1","database_id":database_id,"status":"priced","json_path":json_path.relative_to(PROJECT_ROOT).as_posix(),"generation_script":generator,"summary":summary,"time_grid":analytic_time_grid() if not config.stochastic else exact_transition_time_grid(event_label),"outputs":{"price":"Present value","standard_error":standard_error},**({"exercise":exercise} if exercise else {}),"model_database":model_database,**({"curve_database":curve_database} if curve_database else {}),"product_database":product_database,"result_construction":construction,"timing":timing}
    if config.model_family in {"cir","cir_plus_plus"} and config.stochastic:
        horizon = "first_exercise + (exercise_count - 1) * exercise_period" if config.product_family == "bermudan_swaptions" else "fixing_time" if config.product_family == "caplets" else "expiry"
        spec["time_grid"]={"target_dt":"1/52","step_count":f"round(({horizon}) / target_dt)","effective_dt":f"({horizon}) / step_count"}
    yaml_path.write_text(yaml.safe_dump(spec,sort_keys=False));return json_path
