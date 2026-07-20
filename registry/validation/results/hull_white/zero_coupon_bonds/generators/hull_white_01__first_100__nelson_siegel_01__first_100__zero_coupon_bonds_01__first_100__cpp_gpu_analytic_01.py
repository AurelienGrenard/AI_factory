from pathlib import Path
import sys
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir());sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.result.hull_white.zero_coupon_bonds import generate_result
if __name__=="__main__":print(generate_result(tier="validation",model_id="hull_white_01__first_100",curve_id="nelson_siegel_01__first_100",product_id="zero_coupon_bonds_01__first_100",result_version="01",engine="cpp_gpu_analytic",row_count=100,source_production_result="hull_white_01__nelson_siegel_01__zero_coupon_bonds_01__cpp_gpu_analytic_01"))
