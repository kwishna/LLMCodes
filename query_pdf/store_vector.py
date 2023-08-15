import os

from chromadb import Settings
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

persist_directory = "db"


def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
                texts = text_splitter.split_documents(documents)
                # create embeddings here
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                # create vector store here
                db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory,
                                           client_settings=Settings(
                                               chroma_db_impl='duckdb+parquet',
                                               persist_directory="db",
                                               anonymized_telemetry=False
                                           ))
                db.persist()
                db = None


if __name__ == "__main__":
    main()
