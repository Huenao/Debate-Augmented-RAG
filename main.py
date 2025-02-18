from config import *
from model import *

from flashrag.config import Config
from flashrag.utils import get_dataset


def drag(cfg, test_data):
    pipeline = DebateAugmentedRAG(cfg, max_query_debate_rounds=cfg["max_query_debate_rounds"], max_answer_debate_rounds=cfg["max_answer_debate_rounds"])
    result = pipeline.run(test_data)
    
    return result

def drag_(cfg, test_data):
    pipeline = DebateAugmentedRAG(cfg, max_query_debate_rounds=cfg["max_query_debate_rounds"], max_answer_debate_rounds=cfg["max_answer_debate_rounds"])
    result = pipeline.run(test_data, answer_stage=False)
    
    return result

def drag_2(cfg, test_data):
    pipeline = DebateAugmentedRAG(cfg, max_query_debate_rounds=0, max_answer_debate_rounds=cfg["max_answer_debate_rounds"])
    result = pipeline.run(test_data)
    
    return result

def drag_wo_asys(cfg, test_data):
    pipeline = DebateAugmentedRAG(cfg, max_query_debate_rounds=cfg["max_query_debate_rounds"], max_answer_debate_rounds=cfg["max_answer_debate_rounds"], answer_proponet_agent=2, answer_oppoent_agent=0)
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
        "MAD-RAG": mad_rag,
        "Self-RAG": selfrag,
        "Ret-Robust": retrobust,
        "DRAG": drag,
    }
    
    func = func_map[cfg["method_name"]]
    func(cfg, test_data)
    

if __name__ == "__main__":
    cfg = init_cfg()
    main(cfg)