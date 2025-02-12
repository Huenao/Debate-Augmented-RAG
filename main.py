from config import *
from model import *

from flashrag.config import Config
from flashrag.utils import get_dataset


def drag(cfg, test_data):
    pipeline = DebateAugmentedRAG(cfg, max_debate_rounds=3)
    result = pipeline.run(test_data)
    
    return result

def drag_(cfg, test_data):
    pipeline = DebateAugmentedRAG(cfg, max_debate_rounds=3)
    result = pipeline.run(test_data, answer_stage=False)
    
    return result


def main(cfg):
    all_splits = get_dataset(cfg)
    test_data = all_splits["dev"]
    
    func_map = {
        "Single Agent": single_agent,
        "Standard RAG": standard_rag,
        "FLARE": flare,
        "Iter-RetGen": iterretgen,
        "IRCoT": ircot,
        "Self-Ask": self_ask,
        "SuRe": sure,
        "MAD": mad,
        "MAD-RAG": mad_rag,
        "MAD-RAG2": mad_rag2,
        "Self-RAG": selfrag,
        "Ret-Robust": retrobust,
        "DRAG": drag,
        "DRAG-": drag_
    }
    
    func = func_map[cfg["method_name"]]
    func(cfg, test_data)
    

if __name__ == "__main__":
    cfg = init_cfg()
    main(cfg)