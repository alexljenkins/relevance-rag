from abc import ABC, abstractmethod

class DataStore(ABC):
    # generic base class for all data storage objects
    @abstractmethod
    def add_to_knowledge_store(self, source: str) -> None:
        pass
    
    @abstractmethod
    def retrieve_from_knowledge_store(self, question: str) -> list:
        pass

    @abstractmethod
    def answer_question_with_knowledge(self, question: str) -> str:
        pass
