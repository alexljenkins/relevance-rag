# code 1:1 copy from: https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader
from deepeval.integrations.llama_index import DeepEvalAnswerRelevancyEvaluator

from llama_index.core import download_loader

from llama_index.readers.file import PyMuPDFReader

load_dotenv()

allianz_policy = PyMuPDFReader().load_data(
    file_path="data/POL011BA.pdf", metadata=True
)
nrma_policy = PyMuPDFReader().load_data(
    file_path="data/nrma-car-pds-1023-east.pdf", metadata=True
)
metadata = {'source': 'file', 'alex': 'test'}
# print(nrma_policy[0].metadata)
# print(nrma_policy[1].metadata)
# print(nrma_policy[2].metadata)

metadata += nrma_policy[0].metadata
print(metadata)
from llama_index.core import Document
from llama_index.core.schema import MetadataMode
allianz_policy_text = "\n\n".join([d.get_content() for d in allianz_policy])
nrma_policy_text = "\n\n".join([d.get_content() for d in nrma_policy])

docs = [Document(text=allianz_policy_text), Document(text=nrma_policy_text)]

from llama_index.core.node_parser import HierarchicalNodeParser

node_parser = HierarchicalNodeParser.from_defaults()
nodes = node_parser.get_nodes_from_documents(docs)

from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes
leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)

# define storage context
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from llama_index.llms.openai import OpenAI

docstore = SimpleDocumentStore()

# # insert nodes into docstore
docstore.add_documents(nodes)

# define storage context (will include vector store by default too)
storage_context = StorageContext.from_defaults(docstore=docstore)

# llm = OpenAI(model="gpt-4o")

# Load index into vector index
from llama_index.core import VectorStoreIndex

base_index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)

from llama_index.core.retrievers import AutoMergingRetriever
base_retriever = base_index.as_retriever(similarity_top_k=6)
retriever = AutoMergingRetriever(base_retriever, storage_context, verbose=True)

# query_str = "What were some lessons learned from red-teaming?"
# query_str = "Can you tell me about the key concepts for safety finetuning"
query_str = "I have Third Party Fire & Theft insurance under the allianz policy. What is the most they will repay me for my car?"

nodes = retriever.retrieve(query_str)
base_nodes = base_retriever.retrieve(query_str)

# from llama_index.core.response.notebook_utils import display_source_node

print('Nodes retrieved from AutoMergingRetriever: ')
for node in nodes:
    print(node)

print('BASE Nodes retrieved from AutoMergingRetriever: ')
for node in base_nodes:
    print(node)

from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(retriever)
base_query_engine = RetrieverQueryEngine.from_args(base_retriever)

response = query_engine.query(query_str)

print(str(response))

base_response = base_query_engine.query(query_str)

print(str(base_response))
