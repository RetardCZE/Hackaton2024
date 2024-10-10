"""
TASK 4 - Solution
Make simple vectorstore and implement naive similarity search.

Common gpt models seem like they know a lot of stuff, but in reality, they don't know the stuff, but they have
high probability of completing text with known information as they were trained on it.
Well it is more or less the same in common information. But once you get real specific or into a private data to
which gpt didn't have access during training, you will see that it cannot give you correct answer.

In theory, you could take all your stuff and pass it as context. Which is retarded approach for so many reasons
I won't even expand on it. (Ask GPT why it is dumb I guess you can copy this docstring and ask if I am correct.)

Well what people do is storing all their knowledge in some kind of easily searchable store from which they can
retrieve only relevant subset of their knowledge.

Commonly text content is translated into vectors which are then compared by mathematical comparisons (Like distance).
Your task (4) is to implement such storage from scratch.

"""

from AI_Tutorial.akkodis_clients import client_ada_002
from typing import List

# we will need numpy for some math
import numpy as np
'''
********************************************************************************
TODO: implement simple vector math for similarity criterion
'''
# Implement function to calculate L2 distance.
def l2_distance(vec1: List[float], vec2: List[float]) -> float:
    """Compute L2 distance between 2 vectors
    """
    return 0.0


# Implement function to calculate cosine similarity.
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine between 2 vectors
    """
    return 0.0


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
'''
********************************************************************************
Todo: Generate embeddings for base text and for the query
'''
# Generate embeddings for base strings
# test out how the response looks to know what you need to save
# client.embeddings.create(
#     model=,
#     input=,
#     ...
# )

# Query string
query_text = "Healthy eating and a balanced diet are crucial for well-being"

# Generate embedding for query string
query_embedding: List[float] = [0.0]

# Calculate L2 distances and cosine similarities between query embedding and each base string embedding
distances = []
cosine_similarities = []
for i, base_embedding in ...:
    vector: List[float] = [0.0]  # load from base_embedding (for each sample sentence)
    l2_dist = l2_distance(query_embedding, vector)
    cos_sim = cosine_similarity(query_embedding, vector)
    distances.append((base_texts[i], l2_dist, cos_sim))

'''
********************************************************************************
TODO
'''
# Sort base strings by their L2 distance and cosine similarity to the query string
distances.sort(key=lambda x: x[1])

# Cosine similarities can be sorted in similar manner.
# They are indeed reversed as cos 1 means the angle between the vectors is 0.
# cosine_similarities = sorted(distances, key=, reverse=)

# Display the results
print(f"Query: '{query_text}'")
print("\nBase strings and their L2 distances to the query string:")
for text, l2_dist, _ in distances:
    print(f"'{text}' -> L2 Distance: {l2_dist:.4f}")

print("\nBase strings and their cosine similarities to the query string:")
for text, _, cos_sim in cosine_similarities:
    print(f"'{text}' -> Cosine Similarity: {cos_sim:.4f}")