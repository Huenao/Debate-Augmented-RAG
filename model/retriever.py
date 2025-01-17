from langchain_community.retrievers import WikipediaRetriever
from typing import List
from langchain_core.documents import Document

class MyWikiRetriever:
    def __init__(self, top_k_results: int, doc_content_chars_max: int):
        self.retriever = WikipediaRetriever(top_k_results=top_k_results, doc_content_chars_max=doc_content_chars_max)
    
    def format_docs(self, docs: List[Document]) -> str:
        """Convert Documents to a single string.:"""
        formatted = [
            f"Article Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
            for doc in docs
        ]
        return "\n\n" + "\n\n".join(formatted)
    
    def retrieve(self, query: str) -> str:
        """Retrieve the top k results from Wikipedia."""
        docs = self.retriever.invoke(query)
        return self.format_docs(docs)