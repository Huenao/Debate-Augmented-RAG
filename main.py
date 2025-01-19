from config import *
from model import *

from flashrag.config import Config
from flashrag.utils import get_dataset

def main(cfg):
    all_splits = get_dataset(cfg)
    test_data = all_splits["dev"]
    
    if cfg["method_name"] == "naive_rag":
        naive_rag(cfg, test_data)
    

if __name__ == "__main__":
    cfg = init_cfg()
    main(cfg)