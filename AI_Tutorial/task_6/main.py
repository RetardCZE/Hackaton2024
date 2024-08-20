"""
TASK 6
Implement vectorstore for Gutenberg project Bible to see how it works in large quantities of text.

In task 5 we had just random sentences where it is quite hard to make some validation.
In tasks 6-9 we take Gutenberg Bible project and we will make a bible research assistant.

This specific task is about loading the bible and making simple faiss index for search.
You will save each bible verse separately. When testing the final search you will most likely
get quite bad results as bible verses are commonly quite similar and there is too many of them, making relevant
differentiation nearly impossible.
"""
from AI_Tutorial.akkodis_clients import client_ada_002
import numpy as np
import faiss
import time
import re
from typing import Dict, List, Literal
from pathlib import Path
from .solution import load_bible_dict


'''
********************************************************************************
TODO: implement faiss index and search for top 5 similar items to the query text
As a python exercise you can try to implement loading of gutenberg bible on your own.
But its not AI relevant, so you can remove the function bellow and use

from .solution import load_bible_dict

which will give you what you need. (Or use solution now, but return to this later)
'''
def load_bible_dict() -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Load bible text file to dictionary which you can easily search in.

    Args:
        -

    Returns:
        Nested dictionary with all bible verses in structure:
                        {
                         "Old testament": { bookX: {verseY1: text1,
                                                    verseY2: text2},
                                            bookX2: {...},},
                         "New testament": { bookX3: {verseY3: text3,
                                                     verseY4: text4},
                                            bookX4: {...},},
                        }
    """
    the_bible_dictionary: Dict[str, Dict[str, Dict[str, str]]] = {"Old testament": {},
                                                                  "New testament": {}}
    return the_bible_dictionary


if __name__ == "__main__":
    """
    Embed all the verses and prepare simple similar retrieval as in task 5.
    """
    # We load the bible dictionary and we
    bible_dict = load_bible_dict()

    # prepare embedding client
    client, model = client_ada_002()

    # embed dummy string to get size of embedding vector for faiss index
    # dummy = np.array( ... )

    # Prepare faiss cosine similarity index
    # IndexFlatIP (inner product)

    # Flatten the dictionary to list of texts for embedding and list of metadata so we can access original data
    # from retrieved indices (indices refer to embeddings, but we can keep the information in metadata list)
    metadata: List[Dict[str, str]] = []
    texts = []

   # flatten how?
   # text.append(verse) - verse will be then embedded
   # metadata.append({'testament': testament, 'book': book, 'verse': verse})


    # Embed all verses and add the embeddings to the index in correct order
    # to avoid repetitive embedding which can take quite some time, save all vectors
    # in solution i've saved all as numpy array -  all_vstack.npy (it will work only if we have same flattening
    # to keep metadata the same

    """
    Embedding large quantities can hit ratelimits (Although akkodis api has probably huge rate limits)
    Thats why we should send the embeddings in while loop until succesful with some sleep if failed 
    (to wait out rate limit). Also you can send multiple embeddings at once (batches) to fight network overhead.
    
    while True:
        try:
            # Embed batch
            # add to index
            # break batch loop if success
            break
        except: # exception should be specific
            # wait out rate limitations on fail 
            print("sleeping")
            time.sleep(30)
    """



    # Finally we make user input loop for querying our verse-base (just print what you think is helpful)
    query = ""
    while True:
        query = input("Enter your query: ")
        if query == "quit":
            break
        query_embedding = np.array([client.embeddings.create(
                                    model=model,
                                    input=['query']
                                ).data[0].embedding])
        # D, I = index.search(query_embedding, 10)
        # print
