from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv
from pathlib import Path
import sys


# Enviaroment variables
load_dotenv()


file_path = Path("./semantic search/psallidas_thesis.pdf")

# Load it
loader = PyPDFLoader(file_path)
docs = loader.load()
print("Document page length:", len(docs))

# Split it into chuncks with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print("Total chunks", len(all_splits))

# Create embeddings of the chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vector_store = InMemoryVectorStore(embedding=embeddings)
ids = vector_store.add_documents(documents=all_splits)
print(f"Added {len(ids)} documents to the vector store.")

query = "What does estimator comparison do?"
results = vector_store.similarity_search(query, k=1)

print("Top result:", results[0])
