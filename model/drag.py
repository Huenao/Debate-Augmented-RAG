import string

import Levenshtein
from flashrag.pipeline import BasicPipeline
from flashrag.utils import get_generator, get_retriever
from tqdm import tqdm

from .utils import *


class DebateAugmentedRAG(BasicPipeline):
    def __init__(self, config, prompt_template=None, max_query_debate_rounds=3, max_answer_debate_rounds=3, agents_num=2, proponent_agent=1, opponent_agent=1, generator=None, retriever=None):
        super().__init__(config, prompt_template)
        self.config = config
        self.max_query_debate_rounds = max_query_debate_rounds
        self.max_answer_debate_rounds = max_answer_debate_rounds
        
        if agents_num != proponent_agent + opponent_agent:
            raise ValueError("The number of agents must be equal to the sum of the proponent and opponent agents")

        self.generator = get_generator(config) if generator is None else generator
        self.retriever = get_retriever(config) if retriever is None else retriever
        
        self.agents_messages_answer_stage = dict()
        self.agents_messages_query_stage = dict()
        # Initialize the agents' messages
        for i in range(proponent_agent):
            self.agents_messages_query_stage[f'Proponent Agent {i}'] = []
            self.agents_messages_answer_stage[f'Proponent Agent {i}'] = []
        for i in range(opponent_agent):
            self.agents_messages_query_stage[f'Opponent Agent {i}'] = []
            self.agents_messages_answer_stage[f'Opponent Agent {i}'] = []

    def run(self, dataset, do_eval=True, answer_stage=True):
        for item in tqdm(dataset, desc="Inference: "):
            query_pool = self.query_stage_debate(item)
            item.update_output("QueryStage_QueryPool", query_pool)
            
            if answer_stage:
                self.answer_stage_debate(item, query_pool)
            else:
                message = [
                    self._answer_only_message(query_pool),
                    {"role": "user", "content": f"Question: {item.question}\n"}
                ]
                input_prompt = self.prompt_template.get_string(messages=message)
                output = self.generator.generate(input_prompt)[0]
                item.update_output("answer_input_prompt", input_prompt)
                item.update_output("pred", output)
        
        dataset = self.evaluate(dataset, do_eval=do_eval)
    
    def query_stage_debate(self, item):
        agents_messages = dict()
        query_pool = dict()
        if self.max_query_debate_rounds == 0:
            input_query = item.question
            retrieval_results = self.retriever.search(input_query)
            query_pool[input_query.strip()] = retrieval_results
            return query_pool
        
        for round in range(self.max_query_debate_rounds):
            if round == 0:
                input_query = item.question
                retrieval_results = self.retriever.search(input_query)
                query_pool[input_query.strip()] = retrieval_results
            
            for agent_name in self.agents_messages_query_stage:
                round_message = [
                    self._query_stage_system_message(agent_name),
                    {"role": "user", "content": f"Question: {item.question}\n{self.format_query_pool(query_pool)}"}
                ]
                input_prompt = self.prompt_template.get_string(messages=round_message)
                output = self.generator.generate(input_prompt)[0]

                item.update_output(f"QueryStage_{agent_name}_Round{round}_InputPrompt", input_prompt)
                item.update_output(f"QueryStage_{agent_name}_Round{round}_Output", output)
                
                agents_messages[agent_name] = [input_prompt, output]
            
            moderator_message = [self._query_stage_moderator_message(agents_messages, input_query, query_pool)]
            moderator_input_prompt = self.prompt_template.get_string(messages=moderator_message)
            moderator_output = self.generator.generate(moderator_input_prompt)[0]
            
            item.update_output(f"QueryStage_Moderator_Round{round}_InputPrompt", moderator_input_prompt)
            item.update_output(f"QueryStage_Moderator_Round{round}_Output", moderator_output)
            
            if "Proponent" in moderator_output: # Proponent wins the debate round and the query stage ends
                return query_pool
            else: # Opponent wins the debate round and the query stage continues
                opponent_output = agents_messages["Opponent Agent 0"][1]
                # Update the query pool
                query_pool_tmp = self.maintain_query_pool(query_pool, opponent_output)
                if query_pool is None:
                    return query_pool
                else:
                    query_pool = query_pool_tmp
        
        return query_pool
            
    def answer_stage_debate(self, item, query_pool):
        for round in range(self.max_answer_debate_rounds):
            for agent_name in self.agents_messages_answer_stage:
                if round == 0:
                    init_message = [
                        self._answer_stage_system_message(agent_name, query_pool),
                        {"role": "user", "content": f"Question: {item.question}\n"}
                    ]
                    # initial message
                    self.agents_messages_answer_stage[agent_name] = init_message
                    input_prompt = self.prompt_template.get_string(messages=init_message)
                    output = self.generator.generate(input_prompt)[0]
                
                else:
                    other_agents = {k: v for k, v in self.agents_messages_answer_stage.items() if k != agent_name}
                    debate_message = self._answer_stage_debate_message(other_agents, item.question, round)
                    self.agents_messages_answer_stage[agent_name].append(debate_message)
                    
                    input_prompt = self.prompt_template.get_string(messages=self.agents_messages_answer_stage[agent_name])
                    output = self.generator.generate(input_prompt)[0]
                
                item.update_output(f"AnswerStage_{agent_name}_Round{round}_InputPrompt", input_prompt)
                item.update_output(f"AnswerStage_{agent_name}_Round{round}_Output", output)
                
                self.agents_messages_answer_stage[agent_name].append({"role": "assistant", "content": output})
                
            agents_responses = "Agents responses:\n"
            for agent_name in self.agents_messages_answer_stage:
                agents_responses += f"{agent_name}: {self.agents_messages_answer_stage[agent_name][-1]['content']}\n"
            
            moderator_message = [
                self._answer_stage_moderator_message(),
                {"role": "user", "content": f"Question: {item.question}\n{self.format_query_pool(query_pool)}\n{agents_responses}"}
            ]
            moderator_input_prompt = self.prompt_template.get_string(messages=moderator_message)
            moderator_output = self.generator.generate(moderator_input_prompt)[0]
            item.update_output(f"AnswerStage_Moderator_Round{round}_InputPrompt", moderator_input_prompt)
            item.update_output(f"AnswerStage_Moderator_Round{round}_Output", moderator_output)
            
            item.update_output("pred", moderator_output)
    
    def _query_stage_system_message(self, agent_name):
        if "Proponent" in agent_name:
            system_message = {
                "role": "system",
                "content": "You are a debater. Argue that the current retrieved content is sufficient to answer the question and no further retrieval is needed. Deliver a brief, strong argument with clear reasoning. Do not suggest further retrieval. No extra explanations."
            }        
        elif "Opponent" in agent_name:
            system_message = {
                "role": "system",
                "content": """You are a critical thinker and debater, and your task is to challenge the sufficiency of the current retrieved content. Argue that the current information is insufficient to generate a reliable answer and propose either query optimization or query expansion.
The action you can choose:
1. Query Optimization: If the retrieved content is somewhat relevant but has expression or scope issues, improve the query using this format: Query Optimization: [Original Query] -> [New Query].
2. Query Expansion: If critical information is missing, propose a new query using this format: Query Expansion: [New Query].
Deliver a brief, strong argument with clear reasoning, then you must choose only one action. The output must be in the exact format after your reasoning, without additional explanation, and keep the new query short and precise."""
            }
        else:
            raise ValueError("The agent name must contain either Proponent or Opponent")
        
        return system_message
    
    def _query_stage_moderator_message(self, agents_messages, input_query, query_pool):
        agents_arguments = ""
        for agent in agents_messages:
            agents_arguments += f"{agent}: {agents_messages[agent][1]}\n"
            
        system_message = {
            "role": "system",
            "content": f"You are the judge in a debate. Your task is to evaluate the arguments from agents. There are two types of agents:\n1. Proponent Agent: Argues that the current retrieved content is sufficient.\n2. Opponent Agent: Argues that the current retrieved content is insufficient and proposes query refinement.\nQuestion: {input_query}\n{self.format_query_pool(query_pool)}\nAgents Arguments:\n{agents_arguments}\nOutput only the agent's name. No explanations."
        }
        
        return system_message
    
    def _answer_stage_system_message(self, agent_name, query_pool):
        if "Proponent" in agent_name:
            if self.config["dataset_name"] == "StrategyQA":
                system_message = {
                    "role": "system", 
                    "content": f"Answer the question based on the given document. Given two answer candidates, Yes and No, choose the best answer choice. Explain your answer, and always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response. The following are given documents.\n{self.format_query_pool(query_pool)}"
                }
            else:
                system_message = {
                    "role": "system", 
                    "content": f"Answer the question based on the given document. Explain your answer, and always put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response. The following are given documents.\n{self.format_query_pool(query_pool)}"
                }
        elif "Opponent" in agent_name:
            if self.config["dataset_name"] == "StrategyQA":
                system_message = {
                    "role": "system",
                    "content": "Answer the question based on your own knowledge. Given two answer candidates, Yes and No, choose the best answer choice. Explain your answer, and always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response."
                }
            else:
                system_message = {
                    "role": "system",
                    "content": "Answer the question based on your own knowledge. Explain your answer, and always put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response."
                }
        return system_message
    
    def _answer_only_message(self, query_pool):
        if self.config["dataset_name"] == "StrategyQA":
            system_message = {"role": "system", 
                            "content": f"Answer the question based on the given document. Given two answer candidates, Yes and No, choose the best answer choice. Output only the final answer with no explanations or additional text.\n{self.format_query_pool(query_pool)}"}
        else:
            system_message = {"role": "system", 
                            "content": f"Answer the question based on the given document. Output only the final answer with no explanations or additional text.\n{self.format_query_pool(query_pool)}"}
        return system_message  
    
    def _answer_stage_debate_message(self, other_agents, question, round):
        if self.config["dataset_name"] == "StrategyQA":
            debate_messages = "I will give the answers and arguments to this question from other agents. Use their solution as additional advice; note that they may be wrong. Given two answer candidates, Yes and No, choose the best answer choice. Explain your answer, and always put the answer after 'The answer is: ', e.g.'The answer is: Yes.', at the end of your response."
        else:
            debate_messages = "I will give the answers and arguments to this question from other agents. Use their solution as additional advice; note that they may be wrong. Explain your answer, and always put the answer after 'The answer is: ', e.g.'The answer is: answer.', at the end of your response."
        debate_messages += f"Question: {question}\n"  
        debate_messages += "Other agents responses:\n"
        for i, agent_name in enumerate(other_agents):
            debate_messages += f"{agent_name}: {other_agents[agent_name][round*2]['content']}\n"
        
        return {"role": "user", "content": debate_messages}
    
    def _answer_stage_moderator_message(self):
        if self.config["dataset_name"] == "StrategyQA":
            system_message = {"role": "system", 
                            "content": "You are a moderator in a debate competition. Your task is to determine the correct final answer based on the arguments presented by debaters. Given two answer candidates, Yes and No, choose the best answer choice. Output only the final answer with no explanations or additional text."}
        else:
            system_message = {"role": "system", 
                            "content": "You are a moderator in a debate competition. Your task is to determine the correct final answer based on the arguments presented by debaters. Output only the final answer with no explanations or additional text."}
        return system_message        
    
    def _format_reference(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    
    def maintain_query_pool(self, query_pool, opponent_output):
        try:
            if "Query Optimization:" in opponent_output:
                optimization_instruction = opponent_output.split("Query Optimization:")[1].strip()
                if "->" in optimization_instruction:
                    optimization_instruction = optimization_instruction.split("->")
                    original_query = optimization_instruction[0].strip()
                    new_query = optimization_instruction[1].strip()
                else:
                    original_query = optimization_instruction
                    new_query = optimization_instruction

                 # Remove the original query from the query pool
                query_pool.pop(self.find_most_similar_key(query_pool, original_query))

                retrieval_results = self.retriever.search(new_query)
                query_pool[new_query] = retrieval_results
            elif "Query Expansion:" in opponent_output:
                new_query = opponent_output.split("Query Expansion:")[1].strip()
                retrieval_results = self.retriever.search(new_query)
                query_pool[new_query] = retrieval_results
            else:
                return None
        except Exception as e:
            print(f"Error: {e}")
            print("\n")
            print(query_pool)
            print("\n")
            print(opponent_output)
            raise ValueError("Error in maintaining the query pool")
        
        return query_pool
    
    def format_query_pool(self, query_pool):
        query_pool_str = ""
        for i, query in enumerate(query_pool):
            query_pool_str += f"Query {i+1}: {query}\nRetrieved Content: {self._format_reference(query_pool[query])}"
        
        return query_pool_str
    
    def find_most_similar_key(self, query_dict, target_query):
        min_distance = float('inf')
        most_similar_key = None

        for key in query_dict.keys():
            distance = Levenshtein.distance(key, target_query) # Calculate the Levenshtein distance
            if distance < min_distance:
                min_distance = distance
                most_similar_key = key

        return most_similar_key