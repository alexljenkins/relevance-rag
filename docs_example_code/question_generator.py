# https://docs.llamaindex.ai/en/stable/examples/evaluation/QuestionGeneration/
import logging
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
from llama_index.llms.openai import OpenAI

reader = SimpleDirectoryReader("data/")
documents = reader.load_data()

# data_generator = DatasetGenerator.from_documents(documents)

# eval_questions = data_generator.generate_questions_from_nodes()

# save eval_questions
# df = pd.DataFrame(eval_questions)
# df.to_csv("evaluation_questions.csv", index=False)

# load eval questions as list from lines
eval_questions = []
with open("evaluation_questions.csv", "r") as f:
    for line in f:
        eval_questions.append(line.strip())

print(eval_questions)

gpt4 = OpenAI(temperature=0, model="gpt-4o")
evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)
vector_index = VectorStoreIndex.from_documents(documents)

# loop through questions and create eval results as df

responses = []
eval_results = []
query_engine = vector_index.as_query_engine()

for question in eval_questions:
    response_vector = query_engine.query(question)
    responses.append(response_vector)

    eval_results.append((evaluator_gpt4.evaluate_response(
        query=question, response=response_vector
    )).dict())

# save responses to pickle
import pickle
with open("responses.pkl", "wb") as f:
    pickle.dump(responses, f)
    
with open("eval_results.pkl", "wb") as f:
    pickle.dump(eval_results, f)


# save responses and eval_results to csv
df = pd.DataFrame(responses)
df.to_csv("responses.csv", index=False)

df = pd.DataFrame(eval_results)
df.to_csv("eval_results.csv", index=False)
