from typing import Protocol, Optional
from pydantic import BaseModel, Field


class Generator(Protocol):
    def answer_question_with_context(self, context: list):
        pass


class ContextDataModel(BaseModel):
    knowledge_id: str
    knowledge_context: str
    relevance_score: Optional[float] = None
    source_of_knowledge: Optional[str] = None
    
    @property
    def str_context(self):
        return f"\n## Knowledge ID: {self.knowledge_id}\n## Context Text:\n{self.knowledge_context}\n{'-'*5}"


class AnswerTemplate(BaseModel):
    """ The format of the response given by the AnswerEngine."""
    knowledge_id: str = Field(..., description="The id of the context that the answer is derived from.")
    answer: str = Field(..., description="A concise response to the question, given the knowledge context provided.")
