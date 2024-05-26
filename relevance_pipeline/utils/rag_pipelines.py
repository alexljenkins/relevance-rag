from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from relevance_pipeline.data_stores.store_factory import StorageFactory
from relevance_pipeline.utils.dataloaders import get_pdf_loader
from relevance_pipeline.evaluators.evaluate import EvaluatorType, EvaluatorFactory

def get_inmemory_store(store_name: Literal["bm25", "autovectorstore"] = "autovectorstore"):
    """Specific implementation of a knowledge store to be used as a service."""
    loader = get_pdf_loader()
    store = StorageFactory.get_storage(store_name, loader)

    store.add_to_knowledge_store("data/nrma-car-pds-1023-east.pdf")
    store.add_to_knowledge_store("data/POL011BA.pdf")
    
    # initialise the knowledge store so that it is ready to be queried
    _ = store.retrieve_from_knowledge_store("How many types of insurance policies are there?")
    return store


def get_all_evaluators():
    """Returns all evaluators"""
    return [
        EvaluatorFactory.get_evaluator(EvaluatorType.ANSWER_RELEVANCY),
        EvaluatorFactory.get_evaluator(EvaluatorType.FAITHFULNESS),
        EvaluatorFactory.get_evaluator(EvaluatorType.CONTEXTUAL_RELEVANCY)
    ]

