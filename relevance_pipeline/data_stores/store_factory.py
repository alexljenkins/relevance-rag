from relevance_pipeline.data_stores.store import DataStore
from relevance_pipeline.data_stores.gpt_memory_store import MemoryVectorDataStore


class StorageFactory:
    @staticmethod
    def get_storage(framework: str, loader) -> DataStore:
        """A single interface for creating data storage objects."""
        if framework == "gptmemorystore":
            return MemoryVectorDataStore(loader)
        
        raise TypeError("Invalid data storage selected")
