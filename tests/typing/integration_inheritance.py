import dspy
from dspy.predict.parameter import Parameter
from langchain_core.vectorstores import VectorStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from dynavec.integrations.dspy import DynavecRM
from dynavec.integrations.langchain import DynavecVectorStore
from dynavec.integrations.llamaindex import DynavecLlamaStore


def use_langchain(store: DynavecVectorStore) -> VectorStore:
    store.as_retriever()
    return store


def use_dspy(retriever: DynavecRM) -> dspy.Retrieve:
    retriever.reset()
    retriever.load_state(retriever.dump_state())
    return retriever


def use_dspy_parameter(retriever: DynavecRM) -> Parameter:
    return retriever


def use_llamaindex(store: DynavecLlamaStore) -> BasePydanticVectorStore:
    return store
