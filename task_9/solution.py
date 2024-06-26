"""
TASK 9
Lets add chat history and possibility of reading a specific book.

do list:
    - See how we can use the context for message generation
    - Discussion: how could we improve the answering of the bible chat?
"""

# import openai for the AI stuff, os for importing the key from environment
import openai
import os
import numpy as np
import json
import faiss
import time
import re
from typing import Dict, List, Literal
from pathlib import Path
from utils import load
from openai.types.chat import (ChatCompletionToolMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionUserMessageParam,
                               ChatCompletionAssistantMessageParam,
                               )

'''
****************************************************************************
Step 1:
Load bible text file to some structure which you can easily search in.
my approach:
{testament: { bookX: {verseY: text },},}
'''
index, metadata, the_bible_dictionary, client = load()


'''
****************************************************************************
Step 2:
Define all the tools our bot can use
'''
def search_similar(query: str) -> str:
    query_embedding = np.array([client.embeddings.create(
        model='text-embedding-3-small',
        input=['query']
    ).data[0].embedding])
    D, I = index.search(query_embedding, 10)
    # print(D, I)
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
    return context


def book_list() -> List[str]:
    return list(the_bible_dictionary['Old testament'].keys()) + list(the_bible_dictionary['New testament'].keys())


def book_verses(book: str) -> List[str]:
    if book not in book_list():
        return [f"Book {book} does not exist."]

    testament ='Old testament' if book in list(the_bible_dictionary['Old testament'].keys()) else 'New testament'
    return list(the_bible_dictionary[testament][book].keys())


def read_book_verse(book: str, verse: str) -> str:
    if book not in book_list():
        return f"Book {book} does not exist."
    if verse not in book_verses(book):
        return f"Verse {verse} does not exist in {book}."

    testament ='Old testament' if book in list(the_bible_dictionary['Old testament'].keys()) else 'New testament'
    text = verse + the_bible_dictionary[testament][book][verse]
    return text

'''
****************************************************************************
Step 2.5:
Create list of dictionaries describing the tools
'''
tools = [
{
    'type': 'function',
    'function': {
        'name': 'search_similar',
        "description": "Search for similar parts in the bible (search in Faiss).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query text to search for in the bible.",
                    },
                },
                "required": ["query"],
            },
    }
},
{
    'type': 'function',
    'function': {
        'name': 'book_verses',
        "description": "Get list of all verses in a book",
            "parameters": {
                "type": "object",
                "properties": {
                    "book": {
                        "type": "string",
                        "description": "Name of the book to list verses for",
                    },
                },
                "required": ["book"],
            },
    }
},
{
    'type': 'function',
    'function': {
        'name': 'read_book_verse',
        "description": "Read one verse from a given book from the bible",
            "parameters": {
                "type": "object",
                "properties": {
                    "book": {
                        "type": "string",
                        "description": "The name of the book to read - testament will be found automatically.",
                    },
                    "verse": {
                        "type": "string",
                        "description": "The verse of the book to read (like 1:1, 20:48...)",
                    },
                },
                "required": ["book"],
            },
    }
},
{
    'type': 'function',
    'function': {
        'name': 'book_list',
        "description": "Get list of all books in the bible",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": [],
        },
    }
}

]

'''
****************************************************************************
Step 3:
Make a history aware chatbot able to call your defined functions
Dont forget the system message.
'''
messages: List[ChatCompletionUserMessageParam |
               ChatCompletionSystemMessageParam |
               ChatCompletionAssistantMessageParam |
               ChatCompletionToolMessageParam] = []

system_message = ChatCompletionSystemMessageParam(role='system',
                                                  content="""
You are an expert in Gutenberg bible exploration. You try to answer based on accessible context without
guessing. If you cant find the answer with given tools, say so.                                   
                                                  """)

messages.append(system_message)

query = ""
while True:
    if messages[-1]['role'] == 'assistant' or messages[-1]['role'] == 'system':
        query = input("Enter your message: ")
        if query == "quit":
            break

        message = ChatCompletionUserMessageParam(role='user',
                                                 content=query)
        messages.append(message)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tool_choice='auto',
        tools=tools,
    )
    choice = response.choices[0]
    assistant_msg = ChatCompletionAssistantMessageParam(role='assistant',
                                                        content=choice.message.content,
                                                        tool_calls=choice.message.tool_calls)
    messages.append(assistant_msg)
    if choice.message.content is not None:
        print(choice.message.content)

    if choice.message.tool_calls:
        for tool in choice.message.tool_calls:
            idcall = tool.id
            print(f"Calling {tool.function.name}({tool.function.arguments})")
            response = str(eval(tool.function.name)(**json.loads(tool.function.arguments)))
            tool_msg = ChatCompletionToolMessageParam(role='tool',
                                                      tool_call_id=idcall,
                                                      content=response)
            messages.append(tool_msg)



