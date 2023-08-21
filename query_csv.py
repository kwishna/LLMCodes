import os

import dotenv
import numpy as np
import openai
import pandas as pd
import tiktoken
from numpy import ndarray

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

encoding = tiktoken.get_encoding('gpt2')

df = pd.read_csv("./grades.csv")
# print(df)

df = df.dropna()
# print(df)

df = df[["FirstName", "LastName", "MarksObtained"]]
df['summarized'] = ("first name: " + df['FirstName'].str.strip() + "; last name: " + df[
    'LastName'].str.strip() + "; marks obtained: " + str(df['MarksObtained']))
df['tokens'] = df['summarized'].apply(lambda x: len(encoding.encode(x)))

print(df.head(2))


def get_text_embedding(text, embeddingMode="text-embedding-ada-002"):
    return openai.Embedding.create(input=text, engine=embeddingMode)["data"][0]["embedding"]


def get_df_embedding(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    return {idx: get_text_embedding(r['summarized']) for idx, r in df.iterrows()}


document_embedding = get_df_embedding(df)
print(document_embedding)


def calculate_vector_similarity(x: list[float], y: list[float]) -> ndarray:
    return np.dot(np.array(x), np.array(y))


def get_docs_with_similarity(query: str, df_embedding: dict[tuple[str, str], list[float]]) -> list[
    tuple[ndarray, tuple[str, str]]]:
    query_embedding = get_text_embedding(query)
    document_similarities = sorted(
        [(calculate_vector_similarity(query_embedding, document_embedding), doc_index) for doc_index, document_embedding
         in df_embedding.items()], reverse=True)
    return document_similarities;


print(get_docs_with_similarity("who has scored the highest?", document_embedding))

separator_len = len(encoding.encode("\n* "))


def create_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    relevant_document_sections = get_docs_with_similarity(question, context_embeddings)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    for _, section_index in relevant_document_sections:
        document_section = df.loc[section_index]
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > 500:
            break
        chosen_sections.append("\n* " + document_section.summarized.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))

    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, Say. I don't know.""";
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def get_answer(query: str, df: pd.DataFrame, document_embeddings: dict[(str, str), np.array]) -> str:
    prompt = create_prompt(query, document_embeddings, df)
    response = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=250,
        model="text-davinci-003"
    )
    return response["choices"][0]["text"]


query = "Give me the first name of the student who has scored the highest."
response = get_answer(query, df, document_embedding)
print(f"\nQ: {query}\nA: (response)")
