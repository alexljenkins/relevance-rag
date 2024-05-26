from llama_index.core.readers.base import BaseReader

from relevance_pipeline.data_stores.store import DataStore
from relevance_pipeline.data_stores.automerging_vector_store import MemoryVectorDataStore
from relevance_pipeline.data_stores.bm25_store import MemoryBM25DataStore

class StorageFactory:
    @staticmethod
    def get_storage(framework: str, loader: BaseReader) -> DataStore:
        """A single interface for creating data storage objects."""
        if framework == "autovectorstore":
            return MemoryVectorDataStore(loader)

        elif framework == "bm25":
            return MemoryBM25DataStore(loader)

        # NOTE: implemented a chromadb store, but it didn't play nice with in-memory store
        raise TypeError("Invalid data storage selected")
