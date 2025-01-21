from config import *
from model import *

from flashrag.config import Config
from flashrag.utils import get_dataset

func_map = {
    "Single Agent": single_agent,
    "Standard RAG": standard_rag,
    "FLARE": flare,
    "Iter-RetGen": iterretgen,
    "IRCoT": ircot,
    "Self-Ask": self_ask
}

def main(cfg):
    all_splits = get_dataset(cfg)
    test_data = all_splits["dev"]
    
    func = func_map[cfg["method_name"]]
    func(cfg, test_data)
    

if __name__ == "__main__":
    cfg = init_cfg()
    main(cfg)