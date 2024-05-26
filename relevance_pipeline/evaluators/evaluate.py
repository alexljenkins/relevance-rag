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

"""
TODO: additional evaluators could be:
- speed/latency
- cost and token usage
- context relevance scores
- human eval / ground truth
- generated questions from specific context chunks marking that chunk as the answer block (would require a persistent store though)
    (see: https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern/?h=deepevalanswerrelevancyevaluator#question-generation)
- rank (position) of the answer context block in the retrieved contexts (ie was it the top result or not)
- compare context to what an LLM extracts as the best answer context area - and compare the overlap (since context windows are massive now)
"""
