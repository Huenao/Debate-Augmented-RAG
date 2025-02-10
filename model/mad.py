from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator
from .utils import *

class MultiAgentDebate(BasicPipeline):
    
    def __init__(self, config, prompt_template=None, debate_rounds=3, agents_num=2, use_rag=False, rag_agents_num=1, generator=None, retriever=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.debate_rounds = debate_rounds
        self.agents_num = agents_num-rag_agents_num if use_rag else agents_num # If use rag, the number of agents is the number of agents in the debate minus the number of agents in the RAG

        self.generator = get_generator(config) if generator is None else generator
        if use_rag:
            self.retriever = get_retriever(config) if retriever is None else retriever
        
        self.agents_messages = dict()
        # Initialize the agents' messages
        if use_rag:
            self.rag_agents_num = rag_agents_num
            for i in range(rag_agents_num):
                self.agents_messages[f'RAG Agent {i+1}'] = dict()
        for i in range(agents_num):
            self.agents_messages[f'Agent {i+1}'] = dict()


    def run(self, dataset, do_eval=True, pred_process_fun=single_agent_pred_parse):
        for round in range(self.debate_rounds):
            for i, agent_name in enumerate(self.agents_messages):
                if round == 0:
                    # Construct the system message for the first round
                    if "RAG" in agent_name:
                        input_query = dataset.question
                        # Retrieve the documents
                        retrieval_results = self.retriever.batch_search(input_query)
                        dataset.update_output("retrieval_result", retrieval_results)
                        round_messages = [
                                            [
                                                self._construct_rag_agents_system_message(self._format_reference(r)), # remember to format the reference
                                                {"role": "user", "content": f"Question: {q}"}
                                            ] 
                                            for q, r in zip(dataset.question, dataset.retrieval_result)
                                        ]
                    else:
                        round_messages = [
                                            [
                                                self._construct_agents_system_message(),
                                                {"role": "user", "content": f"Question: {q}"}
                                            ] 
                                            for q in dataset.question
                                        ]
                    # Save the messages
                    self.agents_messages[agent_name] = round_messages
                    # Generate the input prompt
                    input_prompt = [self.prompt_template.get_string(messages=round_message) for round_message in round_messages]
                    # Generate the output
                    output = self.generator.generate(input_prompt)
                    
                    dataset.update_output(f"{agent_name}_Round_{round}_input_prompt", input_prompt)
                    dataset.update_output(f"{agent_name}_Round_{round}_output", output)
                    
                else:
                    for q_id, q in enumerate(dataset.question):
                        other_agents = {k: v for k, v in self.agents_messages.items() if k != agent_name}
                        debate_message = self._construct_debate_user_message(other_agents, q, q_id, round)
                        self.agents_messages[agent_name][q_id].append(debate_message)
                    input_prompt = [self.prompt_template.get_string(messages=agent_message) for agent_message in self.agents_messages[agent_name]]
                    output = self.generator.generate(input_prompt)
                    
                    dataset.update_output(f"{agent_name}_Round_{round}_input_prompt", input_prompt)
                    dataset.update_output(f"{agent_name}_Round_{round}_output", output)
                
                for j, message in enumerate(output):
                    self.agents_messages[agent_name][j].append({"role": "assistant", "content": message})
            
            agents_responses = []
            for i in range(len(dataset.question)):
                str = "Agents responses:\n"
                for agent_name in self.agents_messages:
                    str += f"{agent_name}: {self.agents_messages[agent_name][i][-1]['content']}\n"
                agents_responses.append(str)
                    
            moderator_messages = [
                                    [
                                        self._construct_moderator_system_message(),
                                        {"role": "user", "content": f"Question: {q}\n{agents_responses[i]}"} 
                                    ]
                                    for i, q in enumerate(dataset.question)
                                ]
            moderator_input_prompt = [self.prompt_template.get_string(messages=moderator_message) for moderator_message in moderator_messages]
            moderator_output = self.generator.generate(moderator_input_prompt)
            dataset.update_output(f"Moderator_Round_{round}_input_prompt", moderator_input_prompt)
            dataset.update_output(f"Moderator_Round_{round}_output", moderator_output)

            dataset.update_output("pred", moderator_output)
            
        dataset = self.evaluate(dataset, do_eval=do_eval, pred_process_fun=pred_process_fun)
    
    
    def _construct_agents_system_message(self):
        if self.config["dataset_name"] == "StrategyQA":
            system_message = {"role": "system", 
                              "content": "Answer the question based on your own knowledge. Given two answer candidates, Yes and No, choose the best answer choice. Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response."}
        else:
            system_message = {"role": "system", 
                              "content": "Answer the question based on your own knowledge. Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response."}
        return system_message
    
    def _construct_rag_agents_system_message(self, retrieval_results):
        if self.config["dataset_name"] == "StrategyQA":
            system_message = {"role": "system", 
                              "content": f"Answer the question based on the given document. Given two answer candidates, Yes and No, choose the best answer choice. Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response. The following are given documents.\n{retrieval_results}"}
        else:
            system_message = {"role": "system", 
                              "content": f"Answer the question based on the given document. Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response. The following are given documents.\n{retrieval_results}"}
        return system_message
        
    def _construct_debate_user_message(self, other_agents, question, question_id, round):
        if self.config["dataset_name"] == "StrategyQA":
            debate_messages = "I will give the solution to this question from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer. Given two answer candidates, Yes and No, choose the best answer choice. Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response."
        else:
            debate_messages = "I will give the solution to this question from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer. Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response."
        debate_messages += f"Question: {question}\n"  
        debate_messages += "Other agents responses:\n"
        for i, agent_name in enumerate(other_agents):
            debate_messages += f"{agent_name}: {other_agents[agent_name][question_id][round*2]['content']}\n"
        
        return {"role": "user", "content": debate_messages}
        
    
    def _construct_moderator_system_message(self):
        if self.config["dataset_name"] == "StrategyQA":
            system_message = {"role": "system", 
                            "content": "You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct. The agents' answer always after 'The answer is: '. Also put the agent's answer that you think is correct after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response."}
        else:
            system_message = {"role": "system", 
                            "content": "You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct. The agents' answer always after 'The answer is: '. Also put the agent's answer that you think is correct after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response."}
        return system_message
    
    def _format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
        