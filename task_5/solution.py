"""
TASK 5 - embeddings & RAG
Imagine working on database of 100K documents
do you think your numpy code would be working fast enough? And what about 1M documents and few thousands users?
Let's try some vector databases optimized for fast search.

do list:
    - implement the missing parts of the code
    - compare the speed of your implementation and faiss vectorstore
"""

# import openai for the AI stuff, os for importing the key from environment
import openai
import os
import numpy as np
from texts import base_texts
import faiss
import time


# Function to calculate L2 distance
def l2_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


# Initialize OpenAI client
client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

query_text = "Healthy eating and a balanced diet are crucial for well-being"

# Generate embedding for query string
query_embedding = np.array(client.embeddings.create(
    model='text-embedding-ada-002',
    input=[query_text]
).data[0].embedding)

# Generate embeddings for base strings
base_embeddings = client.embeddings.create(
    model='text-embedding-ada-002',
    input=base_texts
).data
base_embeddings = [emb.embedding for emb in base_embeddings]

index = faiss.IndexFlatL2(query_embedding.shape[0])
index.add(np.array(base_embeddings))

start = time.time()
D, I = index.search(np.array([query_embedding]), 5)
end = time.time()
vectorstore_time = end - start
print("Top 5 by faiss index:", np.array(I[0]), " found in", vectorstore_time, 'time')


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
print(f"For {len(base_embeddings)} items faiss vector store is {numpy_time/vectorstore_time:.2f} times faster than simple numpy approach.")








