from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator
from utils import *

class MultiAgentDebate(BasicPipeline):
    def __init__(self, config, prompt_template=None, debate_rounds=3, agents_num=2):
        super().__init__(config, prompt_template)
        self.debate_rounds = debate_rounds
        self.agents_num = agents_num


    def run(self, dataset, do_eval=True, pred_process_fun=single_agent_pred_parse):
        pass
        