"""
TASK 6 - Solution
Implement vectorstore for Gutenberg project Bible to see how it works in large quantities of text.
"""
from AI_Tutorial.akkodis_clients import client_ada_002
import numpy as np
import faiss
import time
import re
from typing import Dict, List, Literal
from pathlib import Path


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
    # basic init
    bible = Path(__file__).parent / 'bible.txt'
    the_bible_dictionary: Dict[str, Dict[str, Dict[str, str]]] = {"Old testament": {},
                                                                  "New testament": {}}

    current_testament: str = "Old testament"

    current_verse: str = ""
    empty_counter: int = 0

    # open bible file in read mode
    with bible.open('r') as file:
        # read all lines
        lines = file.readlines()

        # we know we begin with Old testament and 1st book title is on line 6 (or zero indexed 5)
        current_book = lines[5].strip()
        the_bible_dictionary[current_testament][current_book] = {}

        # finally we go through all the lines
        for i, line in enumerate(lines[6:]):
            # Between testamets there is *** line which signals that we have to switch to New testament
            if line == '***\n':
                current_testament = '***'
                continue
            if current_testament == '***':
                if line != '\n':
                    current_testament = "New testament"
                continue

            # We have to count empty lines as >= 3 empty lines in a row signalize new book
            if line == '\n':
                empty_counter += 1
                continue

            # If there is large vertical space we know that new book begins
            if empty_counter >= 3:
                current_book = line.strip()
                the_bible_dictionary[current_testament][current_book] = {}
                current_verse = ""
                empty_counter = 0
                continue

            # If there is no new book or testament we are grouping verses in given book
            pattern = r'^\d+:\d+'
            match = re.match(pattern, line)
            if match:
                cut = len(match.group(0))
                current_verse = match.group(0)
                the_bible_dictionary[current_testament][current_book][current_verse] = ''
                text = line.strip()[cut:]
            else:
                text = line.strip()

            if current_verse == '':
                continue

            the_bible_dictionary[current_testament][current_book][current_verse] += text + ' '
            empty_counter = 0
    return the_bible_dictionary


if __name__ == "__main__":
    """
    Embed all the verses and prepare simple similar retrieval as in task 5.
    """
    # We load the bible dictionary and we
    bible_dict = load_bible_dict()

    # prepare client
    client, model = client_ada_002()

    # embed dummy string to get size of embedding vector for faiss index
    dummy = np.array(client.embeddings.create(
        model=model,
        input=['dummy query']
    ).data[0].embedding)

    # Prepare faiss cosine similarity index
    index = faiss.IndexFlatIP(dummy.shape[0])

    # Flatten the dictionary to list of texts for embedding and list of metadata so we can access original data
    # from retrieved indices (indices refer to embeddings, but we can keep the information in metadata list)
    metadata: List[Dict[str, str]] = []
    texts = []
    total = 0

    # nested dict elements counter for progress print
    for testament in bible_dict.keys():
        for book in bible_dict[testament].keys():
            for verse in bible_dict[testament][book].keys():
                total += 1


    # the flattening loops - see how we save the metadata
    progress = 0
    for testament in bible_dict.keys():
        for book in bible_dict[testament].keys():
            for verse in bible_dict[testament][book].keys():
                progress += 1
                print(f'{progress}/{total}')
                texts.append(bible_dict[testament][book][verse])
                metadata.append({'testament': testament, 'book': book, 'verse': verse})


    # Try to load previously embedded data. If not present, make new embeddings for flattened list.
    embeddings_path = Path(__file__).parent / "all_vstack.npy"
    if not embeddings_path.exists():
        print("Generating embeddings...")
        all_embeddings = []
        # here we embed text in batches of 1000 verses
        for i in range(0, len(texts), 1000):
            print(i)
            # As the embedding endpoint can have rate limitations, we upload each batch in loop until successful.
            while True:
                try:
                    embeddings = client.embeddings.create(
                                                          model=model,
                                                          input=texts[i:i + 1000]
                                                         ).data
                    embeddings = np.array([e.embedding for e in embeddings])
                    all_embeddings.append(embeddings)
                    # We load the embeddings to the index as we obtain them.
                    index.add(embeddings)
                    break
                except:
                    print("sleeping")
                    time.sleep(30)

        all_vstack = np.vstack(all_embeddings)
        print( all_vstack.shape)
        np.save('all_vstack.npy', all_vstack)
    else:
        # If we can load saved embeddings, we just load them to faiss index.
        print("Loading embeddings")
        embeddings = np.load('all_vstack.npy')
        index.add(embeddings)

    # Finally we make user input loop for querying our verse-base
    query = ""
    while True:
        query = input("Enter your query: ")
        if query == "quit":
            break
        query_embedding = np.array([client.embeddings.create(
                                    model=model,
                                    input=['query']
                                ).data[0].embedding])
        D, I = index.search(query_embedding, 10)
        for j, i in enumerate(I[0]):
            print(D[0][j], metadata[i])
            print(bible_dict[metadata[i]['testament']][metadata[i]['book']][metadata[i]['verse']])
