# code 1:1 copy from: https://docs.llamaindex.ai/en/stable/examples/retrievers/ensemble_retrieval/
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




from llama_index.core.node_parser import TokenTextSplitter

nodes = TokenTextSplitter(
    chunk_size=1024, chunk_overlap=128
).get_nodes_from_documents(allianz_policy + nrma_policy)


from llama_index.core.storage.docstore import SimpleDocumentStore

docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore("composable", client=client)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(nodes=nodes)
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore, similarity_top_k=2
)

from llama_index.core.schema import IndexNode

vector_obj = IndexNode(
    index_id="vector", obj=vector_retriever, text="Vector Retriever"
)
bm25_obj = IndexNode(
    index_id="bm25", obj=bm25_retriever, text="BM25 Retriever"
)

from llama_index.core import SummaryIndex

summary_index = SummaryIndex(objects=[vector_obj, bm25_obj])

query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize", verbose=True
)

response = query_engine.query(
    "I have Third Party Fire & Theft insurance under the allianz policy. What is the most they will repay me for my car?"
)

print(str(response))