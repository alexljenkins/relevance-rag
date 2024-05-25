# code found: https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern/?h=deepevalanswerrelevancyevaluator#deepeval
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from deepeval.integrations.llama_index import DeepEvalAnswerRelevancyEvaluator


load_dotenv()
# documents = SimpleDirectoryReader("data").load_data()
documents = SimpleWebPageReader(html_to_text=True).load_data([
    "https://www.nrma.com.au/sites/nrma/files/nrma/policy_booklets/nrma-car-pds-1023-east.pdf",
    "https://www.allianz.com.au/openCurrentPolicyDocument/POL011BA/$File/POL011BA.pdf"
])

index = VectorStoreIndex.from_documents(documents)
rag_application = index.as_query_engine()

# An example input to your RAG application
user_input = "What is the policy number for the Allianz policy?"

# LlamaIndex returns a response object that contains
# both the output string and retrieved nodes
response_object = rag_application.query(user_input)

evaluator = DeepEvalAnswerRelevancyEvaluator()
evaluation_result = evaluator.evaluate_response(
    query=user_input, response=response_object
)
print(evaluation_result)


# from deepeval.integrations.llama_index import (
#     DeepEvalAnswerRelevancyEvaluator,
#     DeepEvalFaithfulnessEvaluator,
#     DeepEvalContextualRelevancyEvaluator,
#     DeepEvalSummarizationEvaluator,
#     DeepEvalBiasEvaluator,
#     DeepEvalToxicityEvaluator,
# )