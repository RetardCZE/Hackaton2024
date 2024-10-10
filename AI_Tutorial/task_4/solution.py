"""
TASK 4 - Solution
Make simple vectorstore and implement naive similarity search.
"""

from AI_Tutorial.akkodis_clients import client_ada_002
from typing import List

# we will need numpy for some math
import numpy as np


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


# Function to calculate cosine similarity
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine between 2 vectors

    Args:
        vec1: First input vector.
        vec2: Second input vector (of the same size as vec1).

    Returns:
        Cosine between input vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


# Initialize OpenAI client for embeddings model
client, model = client_ada_002()

# Here is 10 sample sentences to test in embeddings.
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
# the response will contain 10x embedding object (1 for each text)
base_embeddings = client.embeddings.create(
    model=model,
    input=base_texts
).data

# Query string (input for which we want similar texts)
query_text = "Healthy eating and a balanced diet are crucial for well-being"

# Generate embedding for query string
# Here we save directly the list of floats (embedding vector)
query_embedding = client.embeddings.create(
    model=model,
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




