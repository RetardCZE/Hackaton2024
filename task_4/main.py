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
from typing import List


# Function to calculate L2 distance
def l2_distance(vec1: List[float], vec2: List[float]) -> float:
    """one-liner to compute L2 distance between two vectors"""
    pass


# Function to calculate cosine similarity
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
     v1 . v2
     --------
    |v1| . |v2|
    """
    pass


# Initialize OpenAI client
client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

# Knowledgebase
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
# hint: client.embeddings.create
# look to documentation and check how the response looks like
# -> list of embedding vectors
base_embeddings: List[List[float]] = [[]]

# query string (say user input)
query_text = "Healthy eating and a balanced diet are crucial for well-being"

# As for knowledgebase, embed the query
# -> one vector
query_embedding: List[float] = []

# Calculate L2 distances and cosine similarities between query embedding and each base embedding
distances = []
cosine_similarities = []
for i, base_embedding in enumerate(base_embeddings):
    l2_dist = l2_distance(query_embedding, base_embedding)
    cos_sim = cosine_similarity(query_embedding, base_embedding)
    distances.append((base_texts[i], l2_dist, cos_sim))

# Sort base strings by their L2 distance and cosine similarity to the query string
# its just python training, you can look to solution to avoid wasting time (or better use gpt to give you the code)
# *solution is not pretty, it was done also with gpt

# Display the results
print(f"Query: '{query_text}'")
print("\nBase strings and their L2 distances to the query string:")
for text, l2_dist, _ in distances:
    print(f"'{text}' -> L2 Distance: {l2_dist:.4f}")

print("\nBase strings and their cosine similarities to the query string:")
for text, _, cos_sim in cosine_similarities:
    print(f"'{text}' -> Cosine Similarity: {cos_sim:.4f}")




