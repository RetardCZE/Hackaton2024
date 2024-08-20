"""
TASK 7 - Solution
Improve retrieval by grouping small texts.
"""

# import openai for the AI stuff, os for importing the key from environment

import numpy as np
import faiss
import time

from typing import Dict, List, Literal
from pathlib import Path

from AI_Tutorial.akkodis_clients import client_ada_002
from AI_Tutorial.task_6.solution import load_bible_dict


def get_bible_vectorstore():
    """We define this task as function so we can reuse it in Task 8."""
    the_bible_dictionary: Dict[str, Dict[str, Dict[str, str]]] = load_bible_dict()


    # prepare client
    client, model = client_ada_002()

    # make dummy embedding for index size
    dummy = np.array(client.embeddings.create(
        model=model,
        input=['dummy query']
    ).data[0].embedding)

    # we will use cosine similarity search
    index = faiss.IndexFlatIP(dummy.shape[0])

    # We need to flatten the data, but this time we want to merge verses to batches with size near to 2K symbols
    metadata: List[List[Dict[str, str]]] = []
    texts = []
    total = 0

    # nested dict elements counter for progress print
    for testament in the_bible_dictionary.keys():
        for book in the_bible_dictionary[testament].keys():
            for verse in the_bible_dictionary[testament][book].keys():
                total += 1

    # flattening & batching loops
    progress = 0
    current_text = ""
    current_metadata = []
    for testament in the_bible_dictionary.keys():
        for book in the_bible_dictionary[testament].keys():
            for verse in the_bible_dictionary[testament][book].keys():
                # if our current batch has less than 2K symbols we append current verse to it.
                if len(current_text) < 2000:
                    current_text += the_bible_dictionary[testament][book][verse]
                    current_metadata.append({'testament': testament, 'book': book, 'verse': verse,
                                             'length': len(the_bible_dictionary[testament][book][verse])})

                # when we step over 2K symbols, we save current batch as one element for index with its metadata
                # and we reset it to empty container for new verses
                else:
                    texts.append(current_text)
                    metadata.append(current_metadata)
                    current_metadata = []
                    current_text = the_bible_dictionary[testament][book][verse]
                    current_metadata.append({'testament': testament, 'book': book, 'verse': verse,
                                             'length': len(the_bible_dictionary[testament][book][verse])})

    # we append the last batch which probably have not reached 2K symbols
    if len(current_text) > 0:
        texts.append(current_text)
        metadata.append(current_metadata)

    """
    From here its same as Task 6
    we send smaller batches to embeddings as the items are already multiple verses
    also the metadata are different, so printing may be different
    """
    embeddings_path = Path(__file__).parent / "all_vstack.npy"
    if not embeddings_path.exists():
        print("Generating embeddings...")
        all_embeddings = []
        for i in range(0, len(texts), 300):
            print(i)
            while True:
                try:
                    embeddings = client.embeddings.create(
                                                                    model=model,
                                                                    input=texts[i:i + 300]
                                                                ).data
                    embeddings = np.array([e.embedding for e in embeddings])
                    all_embeddings.append(embeddings)
                    index.add(embeddings)
                    break
                except:
                    print("sleeping")
                    time.sleep(30)

        all_vstack = np.vstack(all_embeddings)
        print( all_vstack.shape)
        np.save(embeddings_path, all_vstack)
    else:
        print("Loading embeddings")
        embeddings = np.load(embeddings_path)
        index.add(embeddings)
    return index, metadata, the_bible_dictionary, client, model

if __name__ == "__main__":
    index, metadata, the_bible_dictionary, client, model = get_bible_vectorstore()

    print(index.ntotal)
    query = ""
    while True:
        query = input("Enter your query: ")
        if query == "quit":
            break
        query_embedding = np.array([client.embeddings.create(
                                    model=model,
                                    input=['query']
                                ).data[0].embedding])
        D, I = index.search(query_embedding, 5)
        # NEW - different printing
        # print(D, I)
        for j, i in enumerate(I[0]):
            # print(i, j)
            # print(D[0][j], metadata[i])
            print(80*'-')
            for meta in metadata[i]:
                print(meta['verse'], the_bible_dictionary[meta['testament']][meta['book']][meta['verse']])


