import os
import uuid

import dotenv
import openai
import pandas as pd
import tiktoken

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# contents = []
# tiktoken_embedding = tiktoken.get_encoding('gpt2')
#
# for _file in os.listdir("./store"):
#     with open(f"./store/{_file}", "r") as f:
#         file_content = f.read()
#         tokens = tiktoken_embedding.encode(file_content)
#         token_length = len(tokens)
#         contents.append((_file, file_content, token_length))
#         print(contents)
#
# df = pd.DataFrame(contents, columns=["file", "file_content", "token_length"])
# df["embedding"] = df["file_content"].apply(
#     lambda x: openai.Embedding.create(input=x, engine="text-embedding-ada-002")["data"][0]["embedding"])
#
# df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]  # Generate a unique id for each row in df
# df.to_csv("pinecone.csv")

# -------------------------------------------------------------------

df = pd.read_csv("pinecone.csv")

import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-starter")
index_name = "indexkrishna"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

index = pinecone.Index(index_name)

from tqdm import tqdm

batch_size = 100
chunks = df.to_dict(orient="records")  # Split df into chunks of size batch_size

for i in tqdm(range(0, len(chunks), batch_size)):
    i_end = min(len(chunks), i + batch_size)
    meta_batch = chunks[i:i_end]
    ids_batch = [x["id"] for x in meta_batch]
    embeddings = [x["embedding"] for x in meta_batch]

    data = [{
        "file": x["file"],
        "file_content": x["file_content"]
    } for x in meta_batch]

    _zip = zip(ids_batch, embeddings, data)
    to_upsert = list[_zip]
    # index.upsert(vectors=to_upsert, show_progress=True)

print(index.describe_index_stats())