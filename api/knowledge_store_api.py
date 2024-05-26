import asyncio

from fastapi import FastAPI, Query
from llama_index.core.evaluation.base import BaseEvaluator

from api.api_dataclasses import QAResponse, KnowledgeResponse, AnswerValidationResponse, EvaluationResponse, GeneratorOptionEnum
from relevance_pipeline.utils.rag_pipelines import get_gpt_inmemory_store, get_all_evaluators
from relevance_pipeline.query_ai.open_ai_gpt import create_openai_answer_engine, convert_context_nodes_to_context_model
from relevance_pipeline.evaluators.evaluate import EvaluatorType

app = FastAPI()
BRAIN = get_gpt_inmemory_store()


@app.get("/ask", response_model=QAResponse)
async def ask(question: str):
    """ Answers the question given the information relevant in the knowledge store """
    global BRAIN
    answer = BRAIN.answer_question_with_knowledge(question)
    return QAResponse(question=question, answer=str(answer))


@app.get("/retrieve", response_model=KnowledgeResponse)
async def retrieve(question: str):
    """ Extracts relevant information from the knowledge store """
    global BRAIN
    context_nodes = BRAIN.retrieve_from_knowledge_store(question)
    extracted_data = convert_context_nodes_to_context_model(context_nodes)
    return KnowledgeResponse(question=question, knowledge_context=extracted_data)


@app.get("/validate_answer", response_model=AnswerValidationResponse)
async def validate_retrieve(question: str, generator: GeneratorOptionEnum = Query(..., description="Select a model option")):
    #, eval_method: EvaluatorType = Query(..., description="Select evaluation method")
    """ Extracts relevant information from the knowledge store, uses the information to answer the question and validates the answer """
    global BRAIN
    context_nodes = BRAIN.retrieve_from_knowledge_store(question)
    context_model = convert_context_nodes_to_context_model(context_nodes)

    openai_generator = create_openai_answer_engine(generator.value, temp=0.0)
    answer_model = openai_generator.answer_question_with_context(question, context_model)

    evaluators = get_all_evaluators()
    context_strs = [node.str_context for node in context_model]

    # Run all evaluations in parallel with aevaluate instead of evaluate
    evaluation_tasks = [evaluator.aevaluate(question, answer_model.answer, context_strs) for evaluator in evaluators]
    evaluation_results = await asyncio.gather(*evaluation_tasks)

    evaluations = [
        EvaluationResponse(evaluator=evaluator.__class__.__name__, **eval.dict())
        for evaluator, eval in zip(evaluators, evaluation_results)
    ]

    return AnswerValidationResponse(question = question,
                                    answer = answer_model.answer,
                                    evaluations = evaluations
                                    )
