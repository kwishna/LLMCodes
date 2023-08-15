import os
from langchain import VectorDBQA, OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

loader = DirectoryLoader(path="./dir", glob="**/*.txt")
docs = loader.load()

char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
char_text_splitter.split_documents(documents=docs)

openai_embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPEN_API_KEY'])
vector_store = Chroma.from_documents(documents=docs, embedding=openai_embeddings)

model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff', vectorstore=vector_store)
model.run('what are the effects of homelessness?')

model = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff', vectorstore=vector_store, return_source_documents=True)
res = model({'query': 'what are the effects of homelessness?'})
print(res['source_documents'])