"""Generate the G2++ production model database."""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir())
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.model.core_parameter_databases import write_model_database
from tools.registry.model.random_sampling import philox_uniform_columns
DATABASE_ID="g2_plus_plus_01";ROW_COUNT=1000;SEED=743_200_001
BOUNDS={"mean_reversion_x":(.02,.50),"volatility_x":(.002,.030),"mean_reversion_y":(.05,1.00),"volatility_y":(.002,.030),"rho":(-.95,.50)}
def model_parameters():
    c=philox_uniform_columns(seed=SEED,row_count=ROW_COUNT,bounds=BOUNDS)
    return [{k:float(c[k][i]) for k in BOUNDS} for i in range(ROW_COUNT)]
def main():print(write_model_database(database_id=DATABASE_ID,model_family="g2_plus_plus",parameters=model_parameters(),title="G2++ production model parameter database",construction={"method":"random sample","rule":"The two OU factors and their Brownian correlation are sampled independently within documented bounds.","rng":"Project Philox-4x32-10 generator","seed":SEED,"bounds":{k:list(v) for k,v in BOUNDS.items()}},parameter_docs={"mean_reversion_x":"Positive mean-reversion speed a of the first factor.","volatility_x":"Volatility sigma of the first factor.","mean_reversion_y":"Positive mean-reversion speed b of the second factor.","volatility_y":"Volatility eta of the second factor.","rho":"Brownian correlation between the two factors."},dynamics={"measure":"Risk-neutral","representation":"Deterministically shifted two-factor Gaussian model","equations":{"first_factor":"dx_t = -a x_t dt + sigma dW_t^x","second_factor":"dy_t = -b y_t dt + eta dW_t^y","correlation":"d<W^x,W^y>_t = rho dt","short_rate":"r_t = x_t + y_t + phi(t)","curve_shift":"phi(t) = f(0,t) + 0.5 dV(0,t)/dt"},"initial_conditions":{"first_factor":"x_0 = 0","second_factor":"y_0 = 0"}},registry_tier="production"))
if __name__=="__main__":main()
