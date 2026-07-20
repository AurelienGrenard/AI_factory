from pathlib import Path
import sys
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir());sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.model.slicing import write_model_slice
if __name__=="__main__":print(write_model_slice(project_root=PROJECT_ROOT,source_id="cir_01",target_id="cir_01__first_100",row_count=100))
