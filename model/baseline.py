from .prompt import *
from .utils import *


def single_agent(cfg, test_data):
    from flashrag.pipeline import SequentialPipeline
    # preparation
    templete = single_agent_prompt_template[cfg["dataset_name"]](cfg)
    pipeline = SequentialPipeline(cfg, templete)
    result = pipeline.naive_run(test_data, pred_process_fun=single_agent_pred_parse)
    
    return result


def standard_rag(cfg, test_data):
    from flashrag.pipeline import SequentialPipeline
    # preparation
    templete = standard_rag_prompt_template[cfg["dataset_name"]](cfg)
    pipeline = SequentialPipeline(cfg, templete)
    result = pipeline.run(test_data, pred_process_fun=single_agent_pred_parse)
    
    return result


def flare(cfg, test_data):
    """
    Reference:
        Zhengbao Jiang et al. "Active Retrieval Augmented Generation"
        in EMNLP 2023.
    """
    from flashrag.pipeline import FLAREPipeline

    pipeline = FLAREPipeline(cfg)
    result = pipeline.run(test_data)
    
    return result


def iterretgen(cfg, test_data):
    """
    Reference:
        Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                            Retrieval-Generation Synergy"
        in EMNLP Findings 2023.

        Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
        in EMNLP Findings 2023.
    """
    iter_num = 3

    from flashrag.pipeline import IterativePipeline
    # preparation
    templete = standard_rag_prompt_template[cfg["dataset_name"]](cfg)
    pipeline = IterativePipeline(cfg, prompt_template=templete, iter_num=iter_num)
    result = pipeline.run(test_data, pred_process_fun=single_agent_pred_parse)
    
    return result


def ircot(cfg, test_data):
    """
    Reference:
        Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
        in ACL 2023
    """
    from flashrag.pipeline import IRCOTPipeline

    pipeline = IRCOTPipeline(cfg, max_iter=5)
    result = pipeline.run(test_data)
    
    return result


def self_ask(cfg, test_data):
    """
    Reference:
        Ofir Press et al. "Measuring and Narrowing the Compositionality Gap in Language Models"
        in EMNLP Findings 2023.
    """
    from flashrag.pipeline import SelfAskPipeline

    if cfg["dataset_name"] in ["StrategyQA", "HotpotQA","2wiki"]:
        pipeline = SelfAskPipeline(cfg, max_iter=5, single_hop=True)
        result = pipeline.run(test_data)
    elif cfg["dataset_name"] in ["NQ", "PopQA", "TriviaQA"]:
        pipeline = SelfAskPipeline(cfg, max_iter=5, single_hop=False)
        result = pipeline.run(test_data)
    else:
        raise ValueError("Dataset not supported")
    
    return result


def sure(cfg, test_data):
    """
    Reference:
        Jaehyung Kim et al. "SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs"
        in ICLR 2024
        Official repo: https://github.com/bbuing9/ICLR24_SuRe
    """
    from flashrag.pipeline import SuRePipeline

    pipeline = SuRePipeline(cfg)
    pred_process_fun = lambda x: x.split("\n")[0]
    result = pipeline.run(test_data)
    

def mad(cfg, test_data):
    from .mad import MultiAgentDebate

    pipeline = MultiAgentDebate(cfg, debate_rounds=3, agents_num=2)
    result = pipeline.run(test_data)
    
    return result

def mad_rag(cfg, test_data):
    from .mad import MultiAgentDebate

    pipeline = MultiAgentDebate(cfg, debate_rounds=3, agents_num=2, use_rag=True)
    result = pipeline.run(test_data)
    
    return result


def selfrag(cfg, test_data):
    """
    Reference:
        Akari Asai et al. " SELF-RAG: Learning to Retrieve, Generate and Critique through self-reflection"
        in ICLR 2024.
        Official repo: https://github.com/AkariAsai/self-rag
    """
    from flashrag.pipeline import SelfRAGPipeline

    pipeline = SelfRAGPipeline(
        cfg,
        threshold=0.2,
        max_depth=2,
        beam_width=2,
        w_rel=1.0,
        w_sup=1.0,
        w_use=1.0,
        use_grounding=True,
        use_utility=True,
        use_seqscore=True,
        ignore_cont=True,
        mode="adaptive_retrieval",
    )
    result = pipeline.run(test_data, long_form=False)

    return result


def retrobust(cfg, test_data):
    """
    Reference:
        Ori Yoran et al. "Making Retrieval-Augmented Language Models Robust to Irrelevant Context"
        in ICLR 2024.
        Official repo: https://github.com/oriyor/ret-robust
    """
    model_dict = {
        "nq": "/data/share/baseline_ckpt/Ret_Robust/llama-2-13b-peft-nq-retrobust",
        "2wiki": "/data/share/baseline_ckpt/Ret_Robust/llama-2-13b-peft-2wikihop-retrobust",
    }
    if cfg["dataset_name"] in ["NQ", "TriviaQA", "PopQA", "StrategyQA"]:
        lora_path = model_dict["nq"]
    elif cfg["dataset_name"] in ["HotpotQA", "2wiki"]:
        lora_path = model_dict["2wiki"]
    else:
        print("Not use lora")
        lora_path = model_dict.get(cfg["dataset_name"], None)

    cfg["generator_lora_path"] = lora_path

    from flashrag.pipeline import SelfAskPipeline
    from flashrag.utils import selfask_pred_parse

    pipeline = SelfAskPipeline(cfg, max_iter=5, single_hop=False)
    # use specify prediction parse function
    result = pipeline.run(test_data, pred_process_fun=selfask_pred_parse)

    return result