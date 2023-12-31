"""
Original file is located at
    https://colab.research.google.com/drive/1eFS7_aFaGthiWiIK6y6Hbvsx5TqkpTVz
"""

# !pip -q install langchain huggingface_hub tiktoken
# !pip -q install chromadb
# !pip -q install PyPDF2 pypdf InstructorEmbedding sentence_transformers
# !pip -q install --upgrade together

"""## RetrievalQA with LLaMA 2-70B on Together API"""

import os

os.environ["TOGETHER_API_KEY"] = ""

# !pip show langchain

import together

# set your API key
together.api_key = os.environ["TOGETHER_API_KEY"]

# list available models and descriptons
models = together.Models.list()

together.Models.start("togethercomputer/llama-2-70b-chat")

import logging
from typing import Any, Dict, List, Mapping, Optional

from pydantic import Extra, Field, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text

# !wget -O new_papers_2.zip https://www.dropbox.com/scl/fi/67a80h373n1z38088c9fb/new_papers_2.zip?rlkey=1azfz3w5aazd24ihotwzmol2j&dl=1
# !unzip -q new_papers_2.zip -d new_papers

"""# LangChain multi-doc retriever with ChromaDB

***Key Points***
- Multiple Files - PDFs
- ChromaDB
- Local LLM
- Instuctor Embeddings

## Setting up LangChain
"""

import os

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

"""## Load multiple and process documents"""

# Load and process the text files
# loader = TextLoader('single_text_file.txt')
loader = DirectoryLoader('./new_papers/new_papers/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

len(documents)

#splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

"""## HF Instructor Embeddings"""

from langchain.embeddings import HuggingFaceInstructEmbeddings

instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})

"""## create the DB

This will take a bit of time on a T4 GPU
"""

# Embed and store the texts
# Supplying a persist_directory will store the embeddings on disk

persist_directory = 'db'

## Here is the nmew embeddings being used
embedding = instructor_embeddings

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

"""## Make a retriever"""

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

"""## Make a chain"""

llm = TogetherLLM(
    model= "togethercomputer/llama-2-70b-chat",
    temperature = 0.1,
    max_tokens = 1024
)

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

## Cite sources

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

# full example
query = "What is Flash attention?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

# break it down
query = "What does IO-aware mean?"
llm_response = qa_chain(query)
process_llm_response(llm_response)
# llm_response

query = "What is the context window for LLaMA-2?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "How many tokens was LLaMA-2 trained on?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "When is LLaMA-3 coming?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What is the new model from Meta called?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What is toolformer?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What tools can be used with toolformer?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "How many examples do we need to provide for each tool?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What are the best retrieval augmentations for LLMs?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What is ReAct?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

print(qa_chain.retriever.search_type , qa_chain.retriever.vectorstore)

print(qa_chain.combine_documents_chain.llm_chain.prompt.template)

together.Models.stop("togethercomputer/llama-2-70b-chat")

