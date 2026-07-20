from pathlib import Path
import sys
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir())
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.result.g2_plus_plus.bermudan_swaptions import generate_result
if __name__=="__main__":print(generate_result(tier="production",model_id="g2_plus_plus_01",curve_id="nelson_siegel_01",product_id="bermudan_swaptions_01",result_version="01",engine="cpp_gpu_philox",row_count=1000))
