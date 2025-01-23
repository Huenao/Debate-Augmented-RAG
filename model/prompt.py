from flashrag.prompt import PromptTemplate


single_agent_prompt_template = {
    "StrategyQA" : lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=("Answer the question based on your own knowledge. "
                       "Given four answer candidates, Yes and No, choose the best answer choice. "
                       "Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response. "
                       ),
        user_prompt="Question: {question}",
    ),
}

standard_rag_prompt_template = {
    "StrategyQA" : lambda cfg: PromptTemplate(
        config=cfg,
        system_prompt=("Answer the question based on the given document. "
                       "Given four answer candidates, Yes and No, choose the best answer choice. " 
                       "Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response. "
                       "\nThe following are given documents.\n{reference}"
                       ),
        user_prompt="Question: {question}",
    ),
}