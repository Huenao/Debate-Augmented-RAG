import re

class Agent:
    """
    Agent class for generating responses
    """
    def __init__(self, llm_model, agent_name):
        self.llm_model = llm_model
        self.agent_name = agent_name

        self.agent_contexts = []

    def response(self, message):
        self.agent_contexts.append(message)
        response = self.llm_model.generate_response(self.agent_contexts)
        self.agent_contexts.append(response)

        return response
    
    def clear_contexts(self):
        self.agent_contexts = []


class Agent_with_RAG(Agent):
    """
    Agent with RAG model for retrieving information from Wikipedia
    """
    def __init__(self, llm_model, agent_name, rag_model):
        super().__init__(llm_model, agent_name)
        self.rag_model = rag_model

        self.agent_contexts = []
        self.rag_history = dict()

    def response(self, message):
        self.agent_contexts.append(message)
        response = self.llm_model.generate_response(self.agent_contexts)
        self.agent_contexts.append(response)

        return response

    def rag_result(self, query, rewrite_query=True):
        if rewrite_query:
            query = self._rewrite_query(query)
        retrieved_rst = self.rag_model.retrieve(query)
        self.rag_history[query] = retrieved_rst
        return retrieved_rst
    
    def clear_contexts(self):
        self.agent_contexts = []

    def _parse_query(self, query):
        matches = re.findall(r'"(.*?)"', query)
        return matches[0] if matches else query

    def _rewrite_query(self, question):
        query = f"Here's a question for you to answer, and that question is: {question}\n For this question, what would you search for if given the chance to search from Wikipedia? Just return one of the topics you most want to search for, and put quotation marks around what you want to search for."
        query = self.llm_model.generate_response([{"role": "user", "content": query}])["content"]
        return self._parse_query(query)