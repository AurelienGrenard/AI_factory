from pathlib import Path
import sys
PROJECT_ROOT=next(p for p in Path(__file__).resolve().parents if (p/"registry").is_dir() and (p/"src_cpp").is_dir());sys.path.insert(0,str(PROJECT_ROOT))
from tools.registry.product.slicing import write_product_slice
if __name__=="__main__":print(write_product_slice(project_root=PROJECT_ROOT,source_id="swaptions_01",target_id="swaptions_01__first_100",row_count=100))
