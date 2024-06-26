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


# keep both embeddings as numpy arrays for easier work
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

# look at faiss
# https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/


# implement faiss vectorstore and measure search speed
index = faiss.IndexFlatL2()  # input correct dimensions
index.add()  # load your knowledgebase
index.search()  # find k most similar items to given embedding
vectorstore_time = 0  # measure time

# Calculate L2 distances between query embedding and each base string embedding
# How would you optimize this custom search?
start = time.time()
distances = []
cosine_similarities = []
for i, base_embedding in enumerate(base_embeddings):
    l2_dist = l2_distance(query_embedding, base_embedding)
    distances.append((base_texts[i], l2_dist, i))

# Sort base strings by their L2 distance and cosine similarity to the query string
distances.sort(key=lambda x: x[1])
top_5 = np.array(distances)[:5, 2].astype(int)
end = time.time()
numpy_time = end - start
print("Top 5 by custom similarity search;", top_5, " found in", numpy_time, 'time')
print(f"For {len(base_embeddings)} items faiss vector store is {numpy_time/vectorstore_time:.2f} times faster than simple numpy approach.")








