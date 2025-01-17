from .agent import Agent, Agent_with_RAG
from abc import ABC, abstractmethod

class SingleAgent:
    def __init__(self, llm, args, agent_name="Single Agent"):
        """
        Initialize the SingleAgent class
        :param llm: the language model
        :param args: the arguments
        :param agent_name: the agent name
        """
        self.llm = llm
        self.args = args
        self.agent = Agent(llm_model=self.llm, agent_name=agent_name)
    
    def reason(self, question):
        """
        Start the reasoning
        :param question: the question to be reasoned
        """
        message = self._construct_question_message_for_single_agent(question)

        response = self.agent.response(message)
        print("-"*10)
        print("{} response:\n{}\n".format(self.agent.agent_name, response["content"]))

        return {f"{self.agent.agent_name}": self.agent.agent_contexts}
    
    def _construct_question_message_for_single_agent(self, question):
        """
        Construct the question message for the single agent
        :param question: the question
        """
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
Your final answer should be a single numerical number, and put the answer in the form 'The answer is .', e.g.'The answer is 5.', at the end of your response.
            """.format(question)

        elif self.args.dataset == "ASQA":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
            """.format(question)

        return {"role": "user", "content": question_message}
    
    def clear_contexts(self):
        """
        Clear the contexts of the agent
        """
        self.agent.clear_contexts()


class CoT(SingleAgent):
    def __init__(self, llm, args, agent_name="CoT"):
        super().__init__(llm, args, agent_name)

    def reason(self, question):
        """
        Start the reasoning
        :param question: the question to be reasoned
        """
        message = self._construct_question_message_for_CoT(question)

        response = self.agent.response(message)
        print("-"*10)
        print("{} response:\n{}\n".format(self.agent.agent_name, response["content"]))

        return {f"{self.agent.agent_name}": self.agent.agent_contexts}

    def _construct_question_message_for_CoT(self, question):
        """
        Construct the question message for the single agent
        :param question: the question
        """
        if "MMLU" in self.args.dataset:
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
Let's think step by step.
            """.format(question)

        elif self.args.dataset == "StrategyQA":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
Answer the question with (A) Yes or (B) No. Just select an option and explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A) Yes.', at the beginning of your response.
Let's think step by step.
            """.format(question)

        elif self.args.dataset == "SimpleEthical":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
Let's think step by step.
            """.format(question)
        
        elif self.args.dataset == "GSM8K":
            question_message = """Can you solve the following math problem as accurately as possible?
The question is: {}
Your final answer should be a single numerical number, and put the answer in the form 'The answer is .', e.g.'The answer is 5.', at the end of your response.
Let's think step by step.
            """.format(question)

        elif self.args.dataset == "ASQA":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
Let's think step by step.
            """.format(question)
        
        return {"role": "user", "content": question_message}
    

class Reflection(SingleAgent):
    def __init__(self, llm, args, agent_name="Reflection"):
        super().__init__(llm, args, agent_name)

    def reason(self, question):
        """
        Start the reasoning
        :param question: the question to be reasoned
        """
        for round in range(self.args.rounds):
            print("*******Round {}*******".format(round))
            if round == 0:
                message = self._construct_question_message_for_reflection(question)
            else:
                message = self._construct_reflection_message()

            response = self.agent.response(message)
            print("-"*10)
            print("{} response:\n{}\n".format(self.agent.agent_name, response["content"]))

        return {f"{self.agent.agent_name}": self.agent.agent_contexts}

    def _construct_question_message_for_reflection(self, question):
        """
        Construct the question message for the reflection
        :param question: the question
        """
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
Your final answer should be a single numerical number, and put the answer in the form 'The answer is .', e.g.'The answer is 5.', at the end of your response.
            """.format(question)
        
        elif self.args.dataset == "ASQA":
            question_message = """Can you answer the following question as accurately as possible?
The question is: {}
            """.format(question)
        
        return {"role": "user", "content": question_message}
    
    def _construct_reflection_message(self):
        """
        Construct the reflection message for the reflection
        """
        if "MMLU" in self.args.dataset:
            reflection_message = """Can you double check that your answer is correct.
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """
        
        elif self.args.dataset == "StrategyQA":
            reflection_message = """Can you double check that your answer is correct.
Answer the question with (A) Yes or (B) No. Just select an option and explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A) Yes.', at the beginning of your response.
            """
        
        elif self.args.dataset == "SimpleEthical":
            reflection_message = """Can you double check that your answer is correct.
Explain your answer, put the answer in the form 'The answer is .', e.g.'The answer is (A).', at the beginning of your response.
            """
        
        elif self.args.dataset == "GSM8K":
            reflection_message = """Can you double check that your answer is correct.
Your final answer should be a single numerical number, and put the answer in the form 'The answer is .', e.g.'The answer is 5.', at the end of your response.
            """
        
        elif self.args.dataset == "ASQA":
            reflection_message = """Can you double check that your answer is correct.
            """
            
        return {"role": "user", "content": reflection_message}