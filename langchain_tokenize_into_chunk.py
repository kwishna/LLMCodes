import os

import openai, tiktoken
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain

loader = UnstructuredFileLoader('./dir/data.txt')  # Too many characters
docs = loader.load()

llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])
model = load_summarize_chain(llm=llm, chain_type='stuff')  # Error if token is more than limit.
model.run(docs)

# ------------ ------------- ------------- ------------- ------------- ------------

char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = char_text_splitter.split_documents(docs)

model = load_summarize_chain(llm=llm, chain_type='map_reduce')
model.run(docs)

model = load_summarize_chain(llm=llm, chain_type='refine')
model.run(docs)


# ------------ ------------- ------------- ------------- ------------- ------------

def read_file_create_chunks(fname, chunk_size, overlap):
    tt_encoding = tiktoken.get_encoding('gpt2')

    with open(fname, 'r', encoding='utf-8') as f:
        file_text = f.read()

    tokens = tt_encoding.encode(file_text)
    total_tokens = len(tokens)

    chunks = []

    for i in range(0, total_tokens, chunk_size - overlap):
        chunk = tokens[i: i + chunk_size]
        chunks.append(chunk)

    return chunks


chunks = read_file_create_chunks('./dir', 3000, 50)

openai.api_key = os.environ['OPENAI_API_KEY']

final_response = []
tt_encoding = tiktoken.get_encoding('gpt2')

for index, chunk in enumerate(chunks):
    response = openai.Completion.create(
        model='text-davinci-002',
        prompt=f'Please summarize this: {tt_encoding.decode(chunk)}',
        temperature=0,
        max_tokens=350
    )
    final_response.append(response['choices'][0]['text'])

print(final_response)
