"""
TASK 8
Now we use chat completion for analysis of the chat

do list:
    - See how we can use the context for message generation
    - Discussion: how could we improve the answering of the bible chat?
"""

# import openai for the AI stuff, os for importing the key from environment
import openai
import os
import numpy as np
import faiss
import time
import re
from typing import Dict, List, Literal
from pathlib import Path


'''
****************************************************************************
Step 1:
Load bible text file to some structure which you can easily search in.
my approach:
{testament: { bookX: {verseY: text },},}
'''
bible = Path(__file__).parent / 'bible.txt'
counter = 0
the_bible_dictionary: Dict[str, Dict[str, Dict[str, str]]] = {"Old testament": {},
                                                              "New testament": {}}

LINE_TYPE = Literal['testament', 'book', 'verse', 'empty']
current_testament: str = "Old testament"
current_book: str = ""
current_verse: str = ""
empty_counter: int = 0

# prepare bible to some relevant structure
with bible.open('r') as file:
    lines = file.readlines()
    current_book = lines[5].strip()
    the_bible_dictionary[current_testament][current_book] = {}
    for i, line in enumerate(lines[6:]):
        # division of testaments
        if line == '***\n':
            current_testament = '***'
            continue
        if current_testament == '***':
            if line != '\n':
                current_testament = "New testament"
            continue

        # skipping and counting of empty lines
        if line == '\n':
            empty_counter += 1
            continue

        # getting book titles
        if empty_counter >= 3:
            current_book = line.strip()
            the_bible_dictionary[current_testament][current_book] = {}
            current_verse = ""
            empty_counter = 0
            continue

        # look for verse beginning
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


'''
****************************************************************************
Step 2:
Embed the verses in batches limited by 2000 symbols
'''
# now prepare the vectorstore
# Initialize OpenAI client
client = openai.Client(api_key=os.environ.get('OPENAI_API_KEY'))

dummy = np.array(client.embeddings.create(
    model='text-embedding-3-small',
    input=['dummy query']
).data[0].embedding)

# we will use cosine similarity search
index = faiss.IndexFlatIP(dummy.shape[0])
metadata: List[List[Dict[str, str]]] = []
texts = []
total = 0
for testament in the_bible_dictionary.keys():
    for book in the_bible_dictionary[testament].keys():
        for verse in the_bible_dictionary[testament][book].keys():
            total += 1

progress = 0
current_text = ""
current_metadata = []
for testament in the_bible_dictionary.keys():
    for book in the_bible_dictionary[testament].keys():
        for verse in the_bible_dictionary[testament][book].keys():
            if len(current_text) < 2000:
                current_text += the_bible_dictionary[testament][book][verse]
                current_metadata.append({'testament': testament, 'book': book, 'verse': verse,
                                         'length': len(the_bible_dictionary[testament][book][verse])})
            else:
                texts.append(current_text)
                metadata.append(current_metadata)
                current_metadata = []
                current_text = the_bible_dictionary[testament][book][verse]
                current_metadata.append({'testament': testament, 'book': book, 'verse': verse,
                                         'length': len(the_bible_dictionary[testament][book][verse])})

if len(current_text) > 1500:
    texts.append(current_text)
    metadata.append(current_metadata)

print(len(texts), len(metadata))
embeddings_path = Path(__file__).parent / "all_vstack.npy"
if not embeddings_path.exists():
    print("Generating embeddings...")
    all_embeddings = []
    for i in range(0, len(texts), 300):
        print(i)
        while True:
            try:
                embeddings = client.embeddings.create(
                                                                model='text-embedding-3-small',
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
    np.save('all_vstack.npy', all_vstack)
else:
    print("Loading embeddings")
    embeddings = np.load('all_vstack.npy')
    index.add(embeddings)


'''
****************************************************************************
Step 3:
Use the retrieved context for generated answer - (garbage probably)
'''
print(index.ntotal)
query = ""
while True:
    query = input("Enter your query: ")
    if query == "quit":
        break
    query_embedding = np.array([client.embeddings.create(
                                model='text-embedding-3-small',
                                input=['query']
                                ).data[0].embedding])
    D, I = index.search(query_embedding, 10)
    # print(D, I)
    # NEW - create prompt message from context and send it for completion.
    context = ""
    current_testament = ""
    current_book = ""
    for j, i in enumerate(I[0]):
        for meta in metadata[i]:
            if meta['testament'] != current_testament:
                current_testament = meta['testament']
                context += "\n\nTestament:" + current_testament
            if meta['book'] != current_book:
                current_book = meta['book']
                context += "\n\nBook:" + current_testament

            context += "\n" + meta['verse'] + the_bible_dictionary[meta['testament']][meta['book']][meta['verse']]

    message_content = f"""
    Based on given parts of the Gutenberg bible, try to answer my question: {query}.
    
    The retrieved context:
    {context}
    
    USE ONLY THE CONTEXT. IF THE QUESTION CANNOT BE ANSWERED BY THE CONTEXT, SAY SO.
    Also try to tell me if some of the retrieved books are promising for further investigation regarding given question.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
             'role': 'user',
             'content': message_content,
             }
        ]
    )
    print(response.choices[0].message.content)

