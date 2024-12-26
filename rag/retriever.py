import os

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "What methods was used for model evaluation?"

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5},
)

relevant_docs = retriever.invoke(query)

print("--- Relevant documents ---")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}: {doc}\n")
