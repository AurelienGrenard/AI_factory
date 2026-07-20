from pathlib import Path
import sys
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir());sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.result.cir_plus_plus.zero_coupon_bonds import generate_result
if __name__=="__main__":print(generate_result(tier="production",model_id="cir_plus_plus_01",curve_id="nelson_siegel_01",product_id="zero_coupon_bonds_01",result_version="01",engine="cpp_gpu_analytic",row_count=1000))
