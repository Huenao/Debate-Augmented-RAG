from .agent import Agent, Agent_with_RAG
from .retriever import MyWikiRetriever

class DebateWithRAG:
    def __init__(self, llm, args):
        """
        Initialize the DebateWithRAG class
        :param llm: the language model
        :param args: the arguments
        """
        self.llm = llm
        self.args = args
        self.agents, self.summary_agent = self._init_agents()

    def _init_agents(self):
        """
        Initialize the agents
        """
        agent_list = [Agent(llm_model=self.llm, agent_name=f"Agent {id}") for id in range(self.args.agents-self.args.rag_agents)]
        for i in range(self.args.rag_agents):
            agent_list.append(Agent_with_RAG(llm_model=self.llm,
                                             agent_name="RAG Agent {}".format(i),
                                             rag_model=MyWikiRetriever(top_k_results=self.args.top_k_results, 
                                                            doc_content_chars_max=self.args.doc_content_chars_max)
                                            ))
        if self.args.use_summary_agent:
            summary_agent = Agent(self.llm, "summary_agent")
            return agent_list, summary_agent
        else:
            return agent_list, None
    
    def reason(self, question):
        """
        Start the debate
        :param question: the question to be debated
        """
        for round in range(self.args.rounds):
                print("*******Round {}*******".format(round))
                for i, agent in enumerate(self.agents):
                    if round == 0:
                        # construct the question message for the first round
                        if "RAG Agent" in agent.agent_name:
                            retrieved_rst = agent.rag_result(question, rewrite_query=self.args.rewrite_query)
                            message = self._construct_question_message_for_rag_agent(question, retrieved_rst)
                        else:
                            message = self._construct_question_message_for_vanilla_agent(question)
                    else:
                        # construct the debate message based on the responses from other agents
                        message = self._construct_debate_message(self.agents[:i] + self.agents[i+1:], question, 2 * round - 1)

                    response = agent.response(message)
                    print("-"*10)
                    print("{} response:\n{}\n".format(agent.agent_name, response["content"]))

                # summarize all agent responses
                if self.args.use_summary_agent:
                    summary_message = self._construct_summary_message(question)
                    final_summary = self.summary_agent.response(summary_message)
                    print("-"*10)
                    print("Summary Agent response:\n{}\n".format(final_summary["content"]))

        return self._get_debate_history()

    def _construct_question_message_for_vanilla_agent(self, question):
        if "MMLU" in self.args.dataset:
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question)

        elif self.args.dataset == "StrategyQA":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
Answer the question with (A) Yes or (B) No. Just select an option and explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A) Yes.', at the beginning of your response.
            """.format(question)

        elif self.args.dataset == "SimpleEthical":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question)

        elif self.args.dataset == "GSM8K":
            question_message = """Can you solve the following math problem as accurately as possible?
The question is: {}
Your final answer should be a single numerical number, and put the answer in the form 'The answer is .', e.g.'The answer is 5.', at the beginning of your response.
            """.format(question)

        elif self.args.dataset == "ASQA":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
            """.format(question)
        
        return {"role": "user", "content": question_message}
    
    def _construct_question_message_for_rag_agent(self, question, retrieved_rst):
        if "MMLU" in self.args.dataset:
            question_message = """I will provide you with some information retrieved from Wikipedia. Based on this information, can you answer the following question as accurately as possible? If the retrieved information helps answer the question, please cite the evidence.
The question is: {}
This is the information that may be related to this question from Wikipedia: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question, retrieved_rst)

        elif self.args.dataset == "StrategyQA":
            question_message = """I will provide you with some information retrieved from Wikipedia. Based on this information, can you answer the following question as accurately as possible? If the retrieved information helps answer the question, please cite the evidence.
The question is: {}
This is the information that may be related to this question from Wikipedia: {}
Answer the question with (A) Yes or (B) No. Just select an option and explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A) Yes.', at the beginning of your response.
            """.format(question, retrieved_rst)

        elif self.args.dataset == "SimpleEthical":
            question_message = """I will provide you with some information retrieved from Wikipedia. Based on this information, can you answer the following question as accurately as possible? If the retrieved information helps answer the question, please cite the evidence.
The question is: {}
This is the information that may be related to this question from Wikipedia: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question, retrieved_rst)

        elif self.args.dataset == "GSM8K":
            question_message = """I will provide you with some information retrieved from Wikipedia. Based on this information, can you solve the following math problem as accurately as possible? If the retrieved information helps solve the problem, please cite the evidence.
The question is: {}
This is the information that may be related to this question from Wikipedia: {}
Your final answer should be a single numerical number, and put the answer in the form 'The answer is .', e.g.'The answer is 5.', at the beginning of your response.
            """.format(question, retrieved_rst)

        elif self.args.dataset == "ASQA":
            question_message = """I will provide you with some information retrieved from Wikipedia. Based on this information, can you answer the following question as accurately as possible? If the retrieved information helps answer the question, please cite the evidence.
The question is: {}
This is the information that may be related to this question from Wikipedia: {}
            """.format(question, retrieved_rst)
        
        return {"role": "user", "content": question_message}
    
    def _construct_debate_message(self, agents, question, idx):
        agents_response = ""
        for agent in agents:
            agents_response += """\n{}'s answer is: {}
            """.format(agent.agent_name, agent.agent_contexts[idx]["content"])
        
        if "MMLU" in self.args.dataset:
            debate_message = """I will give the solution to this question from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer.
The question is: {}
These are the answers from other agents: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question, agents_response)

        elif self.args.dataset == "StrategyQA":
            debate_message = """I will give the solution to this question from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer.
The question is: {}
These are the answers from other agents: {}
Answer the question with (A) Yes or (B) No. Just select an option and explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A) Yes.', at the beginning of your response.
            """.format(question, agents_response)

        elif self.args.dataset == "SimpleEthical":
            debate_message = """I will give the solution to this question from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer.
The question is: {}
These are the answers from other agents: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question, agents_response)
        
        elif self.args.dataset == "GSM8K":
            debate_message = """I will give the solution to this math problem from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer.
The question is: {}
These are the answers from other agents: {}
Your final answer should be a single numerical number, and put the answer in the form 'The answer is (answer).', e.g.'The answer is (5).', at the beginning of your response.
            """.format(question, agents_response)

        elif self.args.dataset == "ASQA":
            debate_message = """I will give the solution to this question from other agents. Use their solution as additional advice; note that they may be wrong. If you disagree with the other agents, please give your reasons and answer; otherwise, revise your previous answer.
The question is: {}
These are the answers from other agents: {}
            """.format(question, agents_response)

        return {"role": "user", "content": debate_message}
    
    def _construct_summary_message(self, question):
        agents_responses = ""
        for i, agent in enumerate(self.agents):
            agents_responses += """\n{}'s answer is: {}
            """.format(agent.agent_name, agent.agent_contexts[-1]["content"])

        if "MMLU" in self.args.dataset:
            summary_message_content = """You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct.
The question is: {}
These are the answers from all agents: {}
Put the agent's answer that you think is correct in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question, agents_responses)

        elif self.args.dataset == "StrategyQA":
            summary_message_content = """You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct.
The question is: {}
These are the answers from all agents: {}
Put the agent's answer that you think is correct in the form 'The answer is .', e.g.'The answer is (A) Yes.', at the beginning of your response.
            """.format(question, agents_responses)

        elif self.args.dataset == "SimpleEthical":
            summary_message_content = """You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct.
The question is: {}
These are the answers from all agents: {}
Put the agent's answer that you think is correct in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """.format(question, agents_responses)

        elif self.args.dataset == "GSM8K":
            summary_message_content = """You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct.
The question is: {}
These are the answers from all agents: {}
Your final answer should be a single numerical number, and put the agent's answer that you think is correct in the form 'The answer is .', e.g.'The answer is 5.', at the beginning of your response.
            """.format(question, agents_responses)

        elif self.args.dataset == "ASQA":
            summary_message_content = """You are a moderator. There will be two debaters involved in a debate competition. They will present their answers a question. You must evaluate both sides’ answers and decide which is correct.
The question is: {}
These are the answers from all agents: {}
Just select the agent's answer that you think is correct.
            """.format(question, agents_responses)

        return {"role": "user", "content": summary_message_content}
    
    def _get_debate_history(self):
        # save the agents' contexts
        debate_history = dict()
        for i, agent in enumerate(self.agents):
            debate_history[f"{agent.agent_name}"] = agent.agent_contexts
        if self.args.use_summary_agent:
            debate_history[f"{self.summary_agent.agent_name}"] = self.summary_agent.agent_contexts

        return debate_history
    
    def clear_contexts(self):
        for agent in self.agents:
            agent.clear_contexts()
        self.summary_agent.clear_contexts()