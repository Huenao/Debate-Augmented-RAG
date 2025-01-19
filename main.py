from config import *

from flashrag.config import Config
from flashrag.utils import get_dataset

def main(cfg):
    all_splits = get_dataset(cfg)
    test_data = all_splits["dev"]
    
    print(test_data)
    

if __name__ == "__main__":
    cfg = init_cfg()
    main(cfg)