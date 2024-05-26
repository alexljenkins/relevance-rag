from enum import Enum
from llama_index.core.evaluation.base import BaseEvaluator
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator
)

class EvaluatorType(Enum):
    ANSWER_RELEVANCY = DeepEvalAnswerRelevancyEvaluator
    FAITHFULNESS = DeepEvalFaithfulnessEvaluator
    CONTEXTUAL_RELEVANCY = DeepEvalContextualRelevancyEvaluator

class EvaluatorFactory:
    @staticmethod
    def get_evaluator(evaluator_type: EvaluatorType) -> BaseEvaluator:
        return evaluator_type.value()
