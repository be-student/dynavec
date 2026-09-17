import dspy
from langchain_core.vectorstores import VectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from dynavec.integrations.dspy import DynavecRM
from dynavec.integrations.langchain import DynavecVectorStore
from dynavec.integrations.llamaindex import DynavecLlamaStore


def use_langchain(store: DynavecVectorStore) -> VectorStore:
    store.as_retriever()
    return store


def use_dspy(retriever: DynavecRM) -> dspy.Retrieve:
    return retriever


def use_llamaindex(store: DynavecLlamaStore) -> BasePydanticVectorStore:
    return store
