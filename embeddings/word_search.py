import os
from openai import OpenAI
import dotenv
import numpy as np
import sys
import pandas as pd

dotenv.load_dotenv()

client = OpenAI()

print(
    "Identify the closest word based on cosine similarity to the sum of the embedding vectors of the words passed as input"
)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


model = "text-embedding-ada-002"

# model = "text-embedding-ada-002"
words = sys.argv[1:]

if not words:
    print("Please provide a list of words as input.")
    sys.exit(1)

print(f"Using {words} as input")

embeddings = []
for word in words:
    vector = client.embeddings.create(input=word, model=model).data[0].embedding
    embeddings.append(vector)

n = len(words)
similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        similarity_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

sum_embedding = np.sum(embeddings, axis=0)
similarities_to_sum = [
    cosine_similarity(sum_embedding, embedding) for embedding in embeddings
]

final = pd.DataFrame(similarity_matrix, index=words, columns=words)
print(final)
