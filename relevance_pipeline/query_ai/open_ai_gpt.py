from typing import List

from llama_index.llms.openai import OpenAI
from llama_index.core.prompts.base import PromptTemplate

from relevance_pipeline.query_ai.query_datamodels import Generator, ContextDataModel, AnswerTemplate

class AnswerEngine(Generator):
    def __init__(self, openai_instance: OpenAI):
        self.openai = openai_instance
        self.prompt_template = PromptTemplate("Context:\n{full_context}\n\nGiven the context above, answer the following question."
            "\n\nYour response should be in a json format with the keys 'answer' and 'knowledge_id' where the answer is most relevant."
            "\nIf the answer is not within any of the contexts, do not attempt to answer the question. Instead return the json response {{'knowledge_id': '', 'answer': ''}}"
            "\n\nQuestion: {question}"
            )

    def answer_question_with_context(self, question: str, context: List[ContextDataModel]) -> AnswerTemplate:
        full_context = "".join([node.str_context for node in context])
        answer = self.openai.structured_predict(AnswerTemplate, self.prompt_template, full_context=full_context, question=question)
        return answer


def create_openai_answer_engine(model='gpt-4o', temp=0.0) -> AnswerEngine:
    llm = OpenAI(model=model, temperature=temp)
    return AnswerEngine(llm)

def convert_context_nodes_to_context_model(context_nodes):
    return [ContextDataModel(
            knowledge_id=node.node_id,
            knowledge_context=node.get_content(),
            relevance_score=node.get_score(),
            source_of_knowledge=node.metadata.get('file_path', None)
        )
        for node in context_nodes
    ]
