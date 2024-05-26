from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.readers.base import BaseReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.retrievers import AutoMergingRetriever

from relevance_pipeline.data_stores.store import DataStore


class MemoryVectorDataStore(DataStore):
    """An in memory data store that uses a root and leaf node VectorStoreIndex
    to encode corpus data and retrieve relevant data results."""
    def __init__(self, loader: BaseReader, num_contexts:int = 6) -> None:
        self.filepaths_of_encoded_data = set()
        self.documents = []
        self.num_contexts = num_contexts
        self.loader = loader

        self.knowledge_store_outdated = True
        self.query_engine_outdated = True

    def add_to_knowledge_store(self, source_filepath: str, metadata: dict = {}) -> None:
        if source_filepath in self.filepaths_of_encoded_data:
            # TODO: could change this to a hash of the file contents or something
            print(f"{source_filepath} already encoded.")
            return

        loaded_file = self.loader.load_data(source_filepath, metadata=True)
        loaded_file[0].metadata.pop('source', None)
        doc_text = "\n\n".join([d.get_content() for d in loaded_file])

        self.documents.append(Document(text=doc_text, metadata={**loaded_file[0].metadata, **metadata}))
        self.filepaths_of_encoded_data.add(source_filepath)

        self.knowledge_store_outdated = True
        self.query_engine_outdated = True

    def _rebuild_knowledge_store(self) -> None:
        """Recreates the whole knowledge store from scratch. As this is all in memory, so no persistence to add or remove data."""
        # Parse documents into nodes
        node_parser = HierarchicalNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(self.documents)
        leaf_nodes = get_leaf_nodes(nodes)
        # root_nodes = get_root_nodes(nodes)

        # create storage device
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        context = StorageContext.from_defaults(docstore=docstore)
        
        # NOTE: can use base retriever standalone, or use to compare leaf method improves performance.
        base_retriever = VectorStoreIndex(leaf_nodes, storage_context=context).as_retriever(similarity_top_k=self.num_contexts)
        self.retriever = AutoMergingRetriever(base_retriever, context, verbose=False)

    def _update_knowledge_store(self) -> None:
        if self.knowledge_store_outdated:
            self._rebuild_knowledge_store()
            self.knowledge_store_outdated = False

    def _update_query_engine(self) -> None:
        if self.query_engine_outdated:
            self.knowledge_engine = RetrieverQueryEngine(self.retriever)
            self.query_engine_outdated = False

    def retrieve_from_knowledge_store(self, question: str):
        self._update_knowledge_store()
        return self.retriever.retrieve(question)
    
    def answer_question_with_knowledge(self, question: str):
        # NOTE: This is technically a generator and shouldn't be here
        # it just doesn't naturally separate due to the structure of the vector stores
        self._update_knowledge_store()
        self._update_query_engine()
        return self.knowledge_engine.query(question)
