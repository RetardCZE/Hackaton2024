"""
TASK 5 -
Let's try some vector databases optimized for fast search and compare it with naive implementation.
"""

# import openai for the AI stuff, os for importing the key from environment
from AI_Tutorial.akkodis_clients import client_ada_002
from typing import List
import numpy as np

# Sample texts are provided in separate file and imported as base_texts list.
from texts import base_texts

# Faiss is vectorstore package by meta. It can be installed as faiss-cpu, faiss-gpu (gpu only on linux I think)
import faiss

# We need time package for measurement of approach speed.
import time


# Function to calculate L2 distance
def l2_distance(vec1: List[float], vec2: List[float]) -> float:
    """Compute L2 distance between 2 vectors

    Args:
        vec1: First input vector.
        vec2: Second input vector (of the same size as vec1).

    Returns:
        L2 distance between input vectors.
    """
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


# Initialize OpenAI client for embeddings model
client, model = client_ada_002()

# Our sample input query
query_text = "Healthy eating and a balanced diet are crucial for well-being"

# Generate embedding for the query string
query_embedding = np.array(client.embeddings.create(
    model=model,
    input=[query_text]
).data[0].embedding)

# Generate embeddings for all base strings
base_embeddings = client.embeddings.create(
    model=model,
    input=base_texts
).data
# read only the float vectors (strip higher level objects)
base_embeddings = [emb.embedding for emb in base_embeddings]

# define faiss index for size of used embeddings
index = faiss.IndexFlatL2(query_embedding.shape[0])

# load all base embeddings as our knowledgebase
index.add(np.array(base_embeddings))

# measure how fast we can retrieve 5 most similar items to our sample input query
start = time.time()
D, I = index.search(np.array([query_embedding]), 5)
end = time.time()
vectorstore_time = end - start
print("Top 5 by faiss index:", np.array(I[0]), " found in", vectorstore_time, 'time')

# As in Task 4 we compute similarity with naive approach.
# Calculate L2 distances between query embedding and each base string embedding
start = time.time()
distances = []
cosine_similarities = []
for i, base_embedding in enumerate(base_embeddings):
    l2_dist = l2_distance(query_embedding, base_embedding)
    distances.append((base_texts[i], l2_dist, i))

# Sort base strings by their L2 distance to the query string
distances.sort(key=lambda x: x[1])
top_5 = np.array(distances)[:5, 2].astype(int)
end = time.time()
numpy_time = end - start
print("Top 5 by custom similarity search;", top_5, " found in", numpy_time, 'time')

# in the end we compare top 5 search by faiss and by naive approach
print(f"For {len(base_embeddings)} items faiss vector store is {numpy_time/vectorstore_time:.2f} times faster than simple numpy approach.")








