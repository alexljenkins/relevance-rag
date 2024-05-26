# code from: https://docs.llamaindex.ai/en/stable/module_guides/evaluating/usage_pattern/?h=deepevalanswerrelevancyevaluator#deepeval
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from deepeval.integrations.llama_index import DeepEvalAnswerRelevancyEvaluator

# load_dotenv()

def get_docs_from_folder(folder_path):
    return SimpleDirectoryReader(folder_path).load_data()

def get_docs_from_web(urls:list[str]):
    return SimpleWebPageReader(html_to_text=True).load_data(urls)

def create_index(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

# An example input to your RAG application
# user_input = "What is the policy number for the Allianz policy?"
user_input = "I have Third Party Fire & Theft insurance under the allianz policy. What is the most they will repay me for my car?"

# rag_engine = create_index(get_docs_from_web([
#     "https://www.nrma.com.au/sites/nrma/files/nrma/policy_booklets/nrma-car-pds-1023-east.pdf",
#     "https://www.allianz.com.au/openCurrentPolicyDocument/POL011BA/$File/POL011BA.pdf"
# ]))

# TODO: issue loading 1 of the pdfs from file with this method - try pdf specific loader
rag_engine = create_index(get_docs_from_folder("data"))
# LlamaIndex returns a response object that contains both the output string and retrieved nodes
response_object = rag_engine.query(user_input)


print(response_object)

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
