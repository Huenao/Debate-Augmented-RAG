from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_retriever, get_generator
from .utils import *

class MultiAgentDebate(BasicPipeline):
    
    def __init__(self, config, prompt_template=None, debate_rounds=3, agents_num=2, generator=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.debate_rounds = debate_rounds
        self.agents_num = agents_num

        self.generator = get_generator(config) if generator is None else generator
        
        self.agents_messages = dict()
        for i in range(agents_num):
            self.agents_messages[f'Agent_{i}'] = dict()


    def run(self, dataset, do_eval=True, pred_process_fun=single_agent_pred_parse):
        for round in range(self.debate_rounds):
            for i, agent_name in enumerate(self.agents_messages):
                if round == 0:
                    # Construct the system message for the first round
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
                              "content": "Answer the question based on your own knowledge. Given four answer candidates, Yes and No, choose the best answer choice. Always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response. "}
        else:
            system_message = {"role": "system", 
                              "content": "Answer the question based on your own knowledge. Always put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response. "}
        return system_message
        
    def _construct_debate_user_message(self, other_agents, question, question_id, round):
        debate_messages = "I will give the solution to this question from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer."
        debate_messages += f"Question: {question}\n"  
        debate_messages += "Other agents responses:\n"
        for i, agent_name in enumerate(other_agents):
            debate_messages += f"{agent_name}: {other_agents[agent_name][question_id][round*2]['content']}\n"
        
        return {"role": "user", "content": debate_messages}
        
    
    def _construct_moderator_system_message(self):
        if self.config["dataset_name"] == "StrategyQA":
            system_message = {"role": "system", 
                            "content": "You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct. Put the agent's answer that you think is correct after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response. "}
        else:
            system_message = {"role": "system", 
                            "content": "You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct. Put the agent's answer that you think is correct after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response. "}
        return system_message
        