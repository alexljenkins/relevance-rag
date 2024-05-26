import logging
import random

from dotenv import load_dotenv

from api.api_dataclasses import GeneratorOptionEnum
from relevance_pipeline.utils.rag_pipelines import get_all_evaluators, get_gpt_inmemory_store
from relevance_pipeline.query_ai.open_ai_gpt import create_openai_answer_engine, convert_context_nodes_to_context_model

logging.basicConfig(level=logging.ERROR)
logging.captureWarnings(True)
load_dotenv()

def main():
    question = "I have Third Party Fire & Theft insurance under the allianz policy. What is the most they will repay me for my car?"

    store = get_gpt_inmemory_store()

    context_nodes = store.retrieve_from_knowledge_store(question)
    context_model = convert_context_nodes_to_context_model(context_nodes)

    openai_generator = create_openai_answer_engine(random.choice(list(GeneratorOptionEnum)))
    answer_model = openai_generator.answer_question_with_context(question, context_model)

    evaluators = get_all_evaluators()

    context_strs = [node.str_context for node in context_model]
    evaluations = []
    for evaluator in evaluators:
        eval = evaluator.evaluate(question, answer_model.answer, context_strs)
        evaluations.append(eval)

    return question, context_model, answer_model, evaluations


if __name__ == "__main__":
    q, c, a, e = main()
    print(f'## Question:\n{q}\n')
    print(f'## Context:')
    for node in c:
        print(f'  - {node}')
    print(f'## Answer:\n{a}\n')
    print(f'## Evaluation(s):')
    for ev in e:
        print(f'  - {ev}')
