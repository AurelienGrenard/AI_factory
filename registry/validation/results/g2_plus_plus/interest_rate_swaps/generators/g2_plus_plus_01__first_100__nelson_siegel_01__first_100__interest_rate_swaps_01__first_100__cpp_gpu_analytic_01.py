from pathlib import Path
import sys
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir())
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.result.g2_plus_plus.interest_rate_swaps import generate_result
if __name__=="__main__":print(generate_result(tier="validation",model_id="g2_plus_plus_01__first_100",curve_id="nelson_siegel_01__first_100",product_id="interest_rate_swaps_01__first_100",result_version="01",engine="cpp_gpu_analytic",row_count=100,source_production_result="registry/production/results/g2_plus_plus/interest_rate_swaps/data/g2_plus_plus_01__nelson_siegel_01__interest_rate_swaps_01__cpp_gpu_analytic_01.json"))
