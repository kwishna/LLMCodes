import os
import openai
import dotenv

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("./store/file1.txt", "r") as f:
    text = f.read()

# from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# nltk_text_splitter = NLTKTextSplitter(chunk_size=1000)

nltk_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = nltk_text_splitter.create_documents([text])
texts = nltk_text_splitter.split_text(text)

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(documents=docs, embedding=embedding)
chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever = vectordb.as_retriever(), chain_type="stuff")

query = 'What is Huggingface?'
print(chain.run(query))