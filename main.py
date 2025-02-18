from config import *
from model import *

from flashrag.config import Config
from flashrag.utils import get_dataset


def drag(cfg, test_data):
    pipeline = DebateAugmentedRAG(cfg, 
                                  max_query_debate_rounds=cfg["max_query_debate_rounds"],
                                  max_answer_debate_rounds=cfg["max_answer_debate_rounds"],
                                  query_opponent_agent=cfg["query_opponent_agent"],
                                  query_proponent_agent=cfg["query_proponent_agent"],
                                  answer_opponent_agent=cfg["answer_opponent_agent"],
                                  answer_proponent_agent=cfg["answer_proponent_agent"])
    result = pipeline.run(test_data)
    
    return result

def main(cfg):
    all_splits = get_dataset(cfg)
    test_data = all_splits["dev"]
    
    func_map = {
        "Naive Gen": naive_gen,
        "Naive RAG": naive_rag,
        "FLARE": flare,
        "Iter-RetGen": iterretgen,
        "IRCoT": ircot,
        "Self-Ask": self_ask,
        "SuRe": sure,
        "MAD": mad,
        "Self-RAG": selfrag,
        "Ret-Robust": retrobust,
        "DRAG": drag,
    }
    
    func = func_map[cfg["method_name"]]
    func(cfg, test_data)
    

if __name__ == "__main__":
    cfg = init_cfg()
    main(cfg)