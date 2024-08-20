"""
TASK 8
Now we use chat completion for analysis of the chat
"""

import numpy as np

from AI_Tutorial.akkodis_clients import client_gpt_4o
from AI_Tutorial.task_7.solution import get_bible_vectorstore


# we reuse task 8 to get index, metadata, the dictionary, embedding client and embedding model
index, metadata, the_bible_dictionary, client_ada, model = get_bible_vectorstore()

# this time we will need client for chat completion
client_gpt, gpt_model = client_gpt_4o()

# Once again we make loop for retrieval, but we use the retrieved data as context for chat completion
query = ""
while True:
    query = input("Enter your query: ")
    if query == "quit":
        break
    query_embedding = np.array([client_ada.embeddings.create(
                                model=model,
                                input=[query]
                                ).data[0].embedding])
    D, I = index.search(query_embedding, 10)

    # Here we assemble the retrieved data to a string form which can be used as message to gpt
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

    # This is the message we will send to the GPT
    message_content = f"""
    Based on given parts of the Gutenberg bible, try to answer my question: {query}.
    
    The retrieved context:
    {context}
    
    USE ONLY THE CONTEXT. IF THE QUESTION CANNOT BE ANSWERED BY THE CONTEXT, SAY SO.
    Also try to tell me if some of the retrieved books are promising for further investigation regarding given question.
    """

    # And this is simple request as in Task 1 or 2
    response = client_gpt.chat.completions.create(
        model=gpt_model,
        messages=[
            {
             'role': 'user',
             'content': message_content,
             }
        ]
    )
    print(response.choices[0].message.content)

