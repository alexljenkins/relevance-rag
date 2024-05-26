from llama_index.core.readers.base import BaseReader

from relevance_pipeline.data_stores.store import DataStore
from relevance_pipeline.data_stores.gpt_memory_store import MemoryVectorDataStore


class StorageFactory:
    @staticmethod
    def get_storage(framework: str, loader: BaseReader) -> DataStore:
        """A single interface for creating data storage objects."""
        if framework == "gptmemorystore":
            return MemoryVectorDataStore(loader)

        # NOTE: implemented a chromadb store, but it didn't play nice with in-memory store
        raise TypeError("Invalid data storage selected")
