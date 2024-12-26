import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
)
import sys
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

file_path = Path(sys.argv[1])
file_directory = file_path.parent
persistent_directory = file_directory / "db" / "chroma_db"

if not os.path.exists(persistent_directory):
    os.makedirs(persistent_directory)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    if file_path.suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_path.suffix == ".txt":
        loader = TextLoader(file_path)
    elif file_path.suffix == ".csv":
        loader = CSVLoader(file_path)
    elif file_path.suffix == ".json":
        loader = JSONLoader(file_path)
    elif file_path.suffix == ".html":
        loader = UnstructuredHTMLLoader(file_path)
    elif file_path.suffix == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        raise ValueError(f"File type {file_path.suffix} not supported.")

    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = file_path.name
    avg_char_per_page = int(np.mean([len(doc.page_content) for doc in documents]))

    text_splitter = CharacterTextSplitter(
        chunk_size=avg_char_per_page // 3, chunk_overlap=0
    )
    print(f"Avg chars per page: {avg_char_per_page}")
    docs = text_splitter.split_documents(documents)
    print(f"Embedding {len(docs)} pages...")
    print(f"Creating Chroma database...")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=str(persistent_directory)
    )

else:
    print("Vector store already exists")
