from llama_index.core import StorageContext
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever

from relevance_pipeline.data_stores.store import DataStore
from relevance_pipeline.query_ai.open_ai_gpt import create_openai_answer_engine, convert_context_nodes_to_context_model

class MemoryBM25DataStore(DataStore):
    """An in memory data store that uses a root and leaf node VectorStoreIndex
    to encode corpus data and retrieve relevant data results."""
    def __init__(self, loader: BaseReader, num_contexts:int = 6) -> None:
        self.filepaths_of_encoded_data = set()
        self.documents = []
        self.num_contexts = num_contexts
        self.loader = loader

        self.knowledge_store_outdated = True

    def add_to_knowledge_store(self, source_filepath: str, *args, **kwargs) -> None:
        if source_filepath in self.filepaths_of_encoded_data:
            # TODO: could change this to a hash of the file contents or something
            print(f"{source_filepath} already encoded.")
            return

        self.documents.extend(self.loader.load_data(source_filepath, metadata=True))
        self.filepaths_of_encoded_data.add(source_filepath)
        self.knowledge_store_outdated = True

    def _rebuild_knowledge_store(self) -> None:
        """Recreates the whole knowledge store from scratch. As this is all in memory, so no persistence to add or remove data."""
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(self.documents)
        
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        
        self.retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=self.num_contexts)

    def _update_knowledge_store(self) -> None:
        if self.knowledge_store_outdated:
            self._rebuild_knowledge_store()
            self.knowledge_store_outdated = False

    def retrieve_from_knowledge_store(self, question: str):
        self._update_knowledge_store()
        return self.retriever.retrieve(question)
    
    def answer_question_with_knowledge(self, question: str):
        # TODO: remove this method to keep answer separate from store
        openai_generator = create_openai_answer_engine('gpt-4o', temp=0.1)
        knowledge = convert_context_nodes_to_context_model(self.retrieve_from_knowledge_store(question))
        return openai_generator.answer_question_with_context(question, knowledge)
