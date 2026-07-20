"""Generate the CIR++ production model database."""
from __future__ import annotations
import math
import sys
from pathlib import Path
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir())
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.common.philox import philox_uniforms
from tools.registry.model.core_parameter_databases import write_model_database
DATABASE_ID="cir_plus_plus_01";ROW_COUNT=1000;SEED=742_200_001
def model_parameters():
    u=[philox_uniforms(SEED,ROW_COUNT,stream=i) for i in range(4)];rows=[]
    for i in range(ROW_COUNT):
        kappa=.1+1.9*float(u[0][i]);theta=.005+.115*float(u[1][i]);x0=.001+.119*float(u[2][i]);lower=max(math.sqrt(kappa*theta/5),.01);upper=min(math.sqrt(12*kappa*theta),.30);sigma=lower+(upper-lower)*float(u[3][i]);rows.append({"initial_factor":x0,"kappa":kappa,"theta":theta,"volatility":sigma})
    return rows
def main():print(write_model_database(database_id=DATABASE_ID,model_family="cir_plus_plus",parameters=model_parameters(),title="CIR++ production model parameter database",construction={"method":"conditional random sample","rule":"CIR factor parameters are uniform with conditional volatility bounds controlling the Feller ratio.","rng":"Project Philox-4x32-10 generator","seed":SEED,"bounds":{"initial_factor":[.001,.12],"kappa":[.1,2.0],"theta":[.005,.12],"volatility":["max(sqrt(kappa theta / 5), 0.01)","min(sqrt(12 kappa theta), 0.30)"],"feller_ratio":["1/6",10.0]}},parameter_docs={"initial_factor":"Initial positive CIR factor x0.","kappa":"Positive CIR mean-reversion speed.","theta":"Positive CIR long-run factor level.","volatility":"CIR square-root volatility."},dynamics={"measure":"Risk-neutral","representation":"Deterministically shifted CIR","equations":{"factor":"dx_t = kappa (theta - x_t) dt + volatility sqrt(x_t) dW_t","short_rate":"r_t = x_t + phi(t)","curve_shift":"phi(t) = f(0,t) - f_CIR(0,t)"},"initial_conditions":{"factor":"x_0 = initial_factor"}},registry_tier="production"))
if __name__=="__main__":main()
