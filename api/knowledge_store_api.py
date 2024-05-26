import asyncio

from fastapi import FastAPI, Query

from api.api_dataclasses import QAResponse, KnowledgeResponse, AnswerValidationResponse, EvaluationResponse, GeneratorOptionEnum
from relevance_pipeline.utils.rag_pipelines import get_inmemory_store, get_all_evaluators
from relevance_pipeline.query_ai.open_ai_gpt import create_openai_answer_engine, convert_context_nodes_to_context_model


app = FastAPI()
HIVEMIND = {
    'vector': get_inmemory_store('autovectorstore'),
    'bm25': get_inmemory_store('bm25')
}

@app.get("/ask", response_model=QAResponse)
async def ask(question: str, brain: str = Query('vector', enum=['vector', 'bm25'])):
    """ Answers the question given the information relevant in the knowledge store """
    answer = HIVEMIND[brain].answer_question_with_knowledge(question)
    return QAResponse(question=question, answer=str(answer))


@app.get("/retrieve", response_model=KnowledgeResponse)
async def retrieve(question: str, brain: str = Query('vector', enum=['vector', 'bm25'])):
    """ Extracts relevant information from the knowledge store """
    context_nodes = HIVEMIND[brain].retrieve_from_knowledge_store(question)
    extracted_data = convert_context_nodes_to_context_model(context_nodes)
    return KnowledgeResponse(question=question, knowledge_context=extracted_data)


@app.get("/validate_answer", response_model=AnswerValidationResponse)
async def validate_retrieve(question: str, brain: str = Query('vector', enum=['vector', 'bm25']), generator: GeneratorOptionEnum = Query(..., description="Select a model option")):
    #, eval_method: EvaluatorType = Query(..., description="Select evaluation method")
    """ Extracts relevant information from the knowledge store, uses the information to answer the question and validates the answer """
    context_nodes = HIVEMIND[brain].retrieve_from_knowledge_store(question)
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
