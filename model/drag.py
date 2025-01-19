from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator

class DebateAugmentedRAG(BasicPipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.retriever = get_retriever(cfg)
        self.generator = get_generator(cfg)

    def run(self, test_data):
        return super().run(test_data)