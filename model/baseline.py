from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse

def naive_rag(cfg, test_data):
    from flashrag.pipeline import SequentialPipeline
    # preparation
    pipeline = SequentialPipeline(cfg)
    result = pipeline.run(test_data)
    
    print(result)