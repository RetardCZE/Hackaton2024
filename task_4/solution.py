"""
TASK 4 - embeddings & RAG
Now let's do some RAG stuff

do list:
    - Implement missing parts of the code
    - see the results, try different query
"""

# import openai for the AI stuff, os for importing the key from environment
import openai
import os
import numpy as np


# Function to calculate L2 distance
def l2_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# Initialize OpenAI client
client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

# Generate 10 base strings
base_texts = [
    "Cats are great pets and they love to climb trees",
    "Machine learning models can be very complex",
    "OpenAI creates advanced AI technologies",
    "The sun sets beautifully over the mountains",
    "JavaScript is widely used for web development",
    "Cooking healthy meals can improve your life",
    "The beach is a perfect place to relax",
    "Classical music can be very soothing",
    "A balanced diet is important for health",
    "Exploring new countries can be an adventure"
]

# Generate embeddings for base strings
base_embeddings = client.embeddings.create(
    model='text-embedding-ada-002',
    input=base_texts
).data

# Create a query string
query_text = "Healthy eating and a balanced diet are crucial for well-being"

# Generate embedding for query string
query_embedding = client.embeddings.create(
    model='text-embedding-ada-002',
    input=[query_text]
).data[0].embedding

# Calculate L2 distances and cosine similarities between query embedding and each base string embedding
distances = []
cosine_similarities = []
for i, base_embedding in enumerate(base_embeddings):
    l2_dist = l2_distance(query_embedding, base_embedding.embedding)
    cos_sim = cosine_similarity(query_embedding, base_embedding.embedding)
    distances.append((base_texts[i], l2_dist, cos_sim))

# Sort base strings by their L2 distance and cosine similarity to the query string
distances.sort(key=lambda x: x[1])
cosine_similarities = sorted(distances, key=lambda x: x[2], reverse=True)

# Display the results
print(f"Query: '{query_text}'")
print("\nBase strings and their L2 distances to the query string:")
for text, l2_dist, _ in distances:
    print(f"'{text}' -> L2 Distance: {l2_dist:.4f}")

print("\nBase strings and their cosine similarities to the query string:")
for text, _, cos_sim in cosine_similarities:
    print(f"'{text}' -> Cosine Similarity: {cos_sim:.4f}")




