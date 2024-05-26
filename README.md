# Relevance.ai RAG project

## Objectives

- Create at least 2 retrieval frameworks.
- Create at least 2 generation frameworks
- Create at least 2 evaluation frameworks and compare the two pipelines.

## Goals

- Get it working
- Atomise each retrieval, generation and evaluation object.
- Use interfaces to enable each object to be swapable for another without the pipeline needing to know the specific implementation
- Use factory patterns to separate instantiation and use

To measure if the goals have been achieved, I should be able to have 8 different pipelines using a single retriever, generator and evaluator (2x2x2).
