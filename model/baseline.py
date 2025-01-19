from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse


def single_agent(cfg, test_data):
    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate

    templete = PromptTemplate(
        config=cfg,
        system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt="Question: {question}",
    )
    pipeline = SequentialPipeline(cfg, templete)
    result = pipeline.naive_run(test_data)
    
    return result


def standard_rag(cfg, test_data):
    from flashrag.pipeline import SequentialPipeline
    # preparation
    pipeline = SequentialPipeline(cfg)
    result = pipeline.run(test_data)
    
    return result


def flare(cfg, test_data):
    """
    Reference:
        Zhengbao Jiang et al. "Active Retrieval Augmented Generation"
        in EMNLP 2023.
        Official repo: https://github.com/bbuing9/ICLR24_SuRe

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

    pipeline = IterativePipeline(cfg, iter_num=iter_num)
    result = pipeline.run(test_data)
    
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