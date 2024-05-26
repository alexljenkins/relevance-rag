from enum import Enum

from pydantic import BaseModel

from relevance_pipeline.query_ai.query_datamodels import ContextDataModel

class GeneratorOptionEnum(str, Enum):
    OpenAI_latest = "gpt-4o"
    GPT4 = "gpt-4"
    GPT35TURBO = "gpt-35-turbo-16k"

    
class QAResponse(BaseModel):
    question: str
    answer: str

class KnowledgeResponse(BaseModel):
    question: str
    knowledge_context: list[ContextDataModel]

class EvaluationResponse(BaseModel):
    evaluator: str
    score: float
    passing: bool
    feedback: str
    
class AnswerValidationResponse(BaseModel):
    question: str
    answer: str
    evaluations: list[EvaluationResponse]
