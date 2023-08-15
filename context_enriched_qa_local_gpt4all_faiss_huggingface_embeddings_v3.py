"""
# Context-enhanced question-answering, local version (all-mpnet-base-v2/FAISS/GPT4ALL)
# This notebook provides an end-to-end implementation of context-enhanced completions logic using open source tools which can be run locally:
# - For generating vector representations of text - HuggingFace model [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) is used;
# - Vectors are stored/similarity search is done using [FAISS](https://github.com/facebookresearch/faiss);
# - Text generation is done using [GPT4ALL](https://github.com/nomic-ai/gpt4all).

##1. Prepare the language-generating model (GPT4ALL, 7B parameters version)

gpt4all-lora-quantized.bin didn't work out of the box, and needed to convert it following:
https://gist.github.com/segestic/4822f3765147418fc084a250598c1fe6.

The resulting file (e.g. gpt4all-lora-q-converted.bin) can be stored on a mounted drive, so that the conversion step can be run only once.

1A. If you don't have the converted model (this needs to be run only once)

# -------
# Download the original bin file to the colab's drive
# !wget https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized.bin

# -------
# Download llama.cpp 7B model, install the packages needed for conversion.
# %pip install pyllama
# %pip install transformers
# %pip install pyllamacpp # pygpt4all -- https://github.com/hwchase17/langchain/pull/3837
# !python3 -m llama.download --model_size 7B --folder llama/

# -------
# Convert the source .bin file
# !pyllamacpp-convert-gpt4all ./gpt4all-lora-quantized.bin llama/tokenizer.model ./gpt4all-lora-q-converted.bin

# -------
# Copy the converted model to your drive, so that it can be used later.
# %cp ./gpt4all-lora-q-converted_pygpt4all.bin /content/gdrive/MyDrive/Gpt4allfiles/gpt4all-lora-q-converted.bin

"""
#
#
"""1B. If you have a converted model stored in your drive"""

# Set the path to the converted model from your drive (your path may differ)
GPT4ALL_MODEL_PATH = "/Gpt4allfiles/gpt4all-lora-q-converted.bin"
print(f'''Will be using the model from {GPT4ALL_MODEL_PATH}''')

# Test the model
# !pip install langchain
# !pip install llama-cpp-python

from langchain import PromptTemplate, LLMChain
from langchain.llms import LlamaCpp

llm = LlamaCpp(model_path=GPT4ALL_MODEL_PATH)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is a banana?"

llm_chain.run(question)
#
"""
##2. Import the "contexts".
We'll be using a PDF document with a JavaScript course. In my test the the file js-for-profs.pdf was uploaded from the mounted google drive
"""

# !pip install langchain
# !pip install pypdf

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

contexts_pdf_path = '/MyDrive/Gpt4allfiles/js-for-profs.pdf'

loader = PyPDFLoader(contexts_pdf_path)

pdf_data = loader.load()
# an array of documents [Document(page_content='some string', metadata={'source': '/MyDrive/Gpt4allfiles/js-for-profs.pdf', 'page': 0})]

print(len(pdf_data))

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=250, chunk_overlap=0, length_function=len
)  # my gpt4all model has token limit of 512, need to fit it

texts = text_splitter.split_documents(pdf_data)

print(f'Now you have {len(texts)} documents')
#
"""##3. Prepare the embeddings model
Will be used by langchain.embeddings.HuggingFaceEmbeddings(). We will be using the model 'all-mpnet-base-v2' (768 dimensions, ~420Mb), which can be saved to our mounted drive to avoid downloading it each time.

3A. If you don't have the embeddings model saved in your gdrive yet
"""
from sentence_transformers import SentenceTransformer

embeddings_model = SentenceTransformer('all-mpnet-base-v2')
embeddings_model_path = 'all-mpnet-base-v2'

# Save the model
embeddings_model.save(embeddings_model_path)
print(f'''Downloaded {embeddings_model_path}''')

# %cp -r all-mpnet-base-v2 /MyDrive/Gpt4allfiles/all-mpnet-base-v2
#
"""3B. If you have the embeddings model mounted already
SequenceTransformer will search the weights from the directory SENTENCE_TRANSFORMERS_HOME,
if the weights are not found, it will download the weights from huggingface hub (https://github.com/hwchase17/langchain/issues/3079)
"""
import os
from sentence_transformers import SentenceTransformer

EMBEDDINGS_MODEL_PATH = "/MyDrive/Gpt4allfiles/all-mpnet-base-v2"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = EMBEDDINGS_MODEL_PATH

embeddings_model = SentenceTransformer()
print(EMBEDDINGS_MODEL_PATH)
#
"""##4. [Vectors storage](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/faiss.html) preparation, generating and storing vectors"""

# !pip install faiss-cpu
# !pip install langchain
from langchain.vectorstores import FAISS
"""4A. If you don't have the index yet - run this once to populate it and save to the drive"""

import os
from langchain.embeddings import HuggingFaceEmbeddings

EMBEDDINGS_MODEL_PATH = "/MyDrive/Gpt4allfiles/all-mpnet-base-v2"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = EMBEDDINGS_MODEL_PATH

faiss_index_name = 'js-profs-faiss-index-250'

faiss_index = FAISS.from_documents(pdf_data, HuggingFaceEmbeddings())

# Save the index locally (not to generate it every time)
faiss_index.save_local(faiss_index_name)

# Copy the index to your drive
# %cp -r js-profs-faiss-index-250 /MyDrive/Gpt4allfiles/js-profs-faiss-index-250
#
"""4B. If you have the index in your drive already"""

import os
from langchain.embeddings import HuggingFaceEmbeddings

EMBEDDINGS_MODEL_PATH = "/MyDrive/Gpt4allfiles/all-mpnet-base-v2"
os.environ['SENTENCE_TRANSFORMERS_HOME'] = EMBEDDINGS_MODEL_PATH

FAISS_INDEX_PATH = "/MyDrive/Gpt4allfiles/js-profs-faiss-index-250"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
db = FAISS.load_local(FAISS_INDEX_PATH,
                      HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME))

# Test the similarity search
embeddings_test_query = "What are constants?"

docs = db.similarity_search(embeddings_test_query, k=1)

for doc in docs:
  print(str(doc.metadata["page"]) + ":", doc.page_content)
  print(f'''>>> Page content:\n\n{docs[0].page_content}''')
  print(f'''>>> Metadata:{docs[0].metadata}''')
  print(f'''>>> Total docs: {len(docs)}''')

docs_and_scores = db.similarity_search_with_score(embeddings_test_query)

print(f'''>>> Similarity score of the 1st doc: {docs_and_scores[0][1]}''')
#
"""##5. Combining everything together"""
# !pip install langchain
# !pip install faiss-cpu
# !pip install sentence_transformers
# !pip install llama-cpp-python

import os
from langchain import LLMChain, PromptTemplate
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

MOUNTED_GDRIVE_FOLDER_PATH = "/MyDrive/Gpt4allfiles/"
EMBEDDINGS_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDINGS_MODEL_PATH = f"{MOUNTED_GDRIVE_FOLDER_PATH}{EMBEDDINGS_MODEL_NAME}"

os.environ['SENTENCE_TRANSFORMERS_HOME'] = EMBEDDINGS_MODEL_PATH
print(f">> Mounted the embeddings model {EMBEDDINGS_MODEL_PATH}")

GPT4ALL_MODEL_NAME = "gpt4all-lora-q-converted.bin"
GPT4ALL_MODEL_PATH = f"{MOUNTED_GDRIVE_FOLDER_PATH}{GPT4ALL_MODEL_NAME}"
print(f">> Mounted the LLM {GPT4ALL_MODEL_PATH}")

FAISS_INDEX_NAME = "js-profs-faiss-index-250"
FAISS_INDEX_PATH = f"{MOUNTED_GDRIVE_FOLDER_PATH}{FAISS_INDEX_NAME}"
print(f">> Mounted the FAISS index {FAISS_INDEX_PATH}")

llm = LlamaCpp(model_path=GPT4ALL_MODEL_PATH)
db = FAISS.load_local(FAISS_INDEX_PATH, HuggingFaceEmbeddings())

# # Prepare the prompt template
template = """Respond to the question based on the context.

Question:
{question}

Context:
{context}"""
prompt = PromptTemplate(template=template,
                        input_variables=["question", "context"])

# Prepare the chain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Retrieve the texts to be used as a context for answering the question
query = 'what is the difference between null and undefined?'
print(f">> Query: {query}")
contexts_list = db.similarity_search(query, k=1)
print(f'>> Contexts list: {contexts_list}')

context = contexts_list[0].page_content
print(f'context: {context}, len={len(context)}')

# Run the chain to generate the answer
llm_chain.run({'question': query, 'context': context[0:300]})
