from pathlib import Path
import sys
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir())
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.result.hull_white.bermudan_swaptions import generate_result
if __name__=="__main__":print(generate_result(tier="validation",model_id="hull_white_01__first_100",curve_id="nelson_siegel_01__first_100",product_id="bermudan_swaptions_01__first_100",result_version="01",engine="python_cpu_pytorch",row_count=100,source_production_result="hull_white_01__nelson_siegel_01__bermudan_swaptions_01__cpp_gpu_philox_01"))
