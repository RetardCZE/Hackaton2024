"""
TASK 9 - Solution
Lets make a full chatbot with tools and such.

Read the code. Play with the queries. Make custom tools.
This is final waypoint before you design your custom AI assistant.
"""

import numpy as np
import json

from typing import Dict, List, Literal

from AI_Tutorial.akkodis_clients import client_gpt_4o
from AI_Tutorial.task_7.solution import get_bible_vectorstore
from openai.types.chat import (ChatCompletionToolMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionUserMessageParam,
                               ChatCompletionAssistantMessageParam,
                               )


# load all needed stuff from task 7 as in task 8
index, metadata, the_bible_dictionary, client_ada, model_ada = get_bible_vectorstore()

# this time we will need client for chat completion
client_gpt, gpt_model = client_gpt_4o()


# Now we define tools that our chatbot can use. You know first tool from other tasks and the rest is simple.
def search_similar(query: str) -> str:
    """ Make a similarity search in your vectorstore and assemble it into a context string

    Args:
        query: A string for which we want to retrieve most similar parts of the bible.

    Returns:
        A context string with retrieved parts of the bible.
    """

    query_embedding = np.array([client_ada.embeddings.create(
        model=model_ada,
        input=[query]
    ).data[0].embedding])
    D, I = index.search(query_embedding, 10)
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
    """ Return list of all books in the bible (something like chapters)

    Args:
        None

    Returns:
        A list of book names in the bible (both testaments)
    """
    return list(the_bible_dictionary['Old testament'].keys()) + list(the_bible_dictionary['New testament'].keys())


def book_verses(book: str) -> List[str]:
    """ Return list of verse numbers for a given book.

    Args:
        book: Name of a book for which we want list of verses (name will be validated in the function)

    Returns:
        List of verse numbers if book exists.
    """
    if book not in book_list():
        return [f"Book {book} does not exist."]

    testament ='Old testament' if book in list(the_bible_dictionary['Old testament'].keys()) else 'New testament'
    return list(the_bible_dictionary[testament][book].keys())


def read_book_verse(book: str, verse: str) -> str:
    """ Read a specific verse from a specific book (verse number is given as number)

    Args:
        book: Name of a book from which we want to read.
        verse: Number of the verse as string (i.e. 6:22) which we want to read.

    """
    if book not in book_list():
        return f"Book {book} does not exist."
    if verse not in book_verses(book):
        return f"Verse {verse} does not exist in {book}."

    testament ='Old testament' if book in list(the_bible_dictionary['Old testament'].keys()) else 'New testament'
    text = verse + the_bible_dictionary[testament][book][verse]
    return text


# Now we have to create a list of all tools that gpt can call.
# Look at openai documentation about tool calls.
# Generally you define function name and description, its arguments, their types and description.
# GPT can then create a request for your side to call that function
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
                        "description": "A single verse of the book to read (i.e 16:24). Range of verses is not supported",
                    },
                },
                "required": ["book", "verse"],
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

# define conversation container as in task 3
messages: List[ChatCompletionUserMessageParam |
               ChatCompletionSystemMessageParam |
               ChatCompletionAssistantMessageParam |
               ChatCompletionToolMessageParam] = []


# Prepare system message for your bible agent
system_message = ChatCompletionSystemMessageParam(role='system',
                                                  content="""
You are an expert in Gutenberg bible exploration. You try to answer based on accessible context without
guessing. If you cant find the answer with given tools, say so.                                   
                                                  """)

messages.append(system_message)

# Here we define query loop similar to Task 3
# the difference is that we give gpt a set of tools which can be used. Due to that we have to check responses for
# such tool calls.
query = ""
while True:

    # We wait for user input if last message was ai or system message
    # if last message was tool response, we just send the conversation to gpt so it gets response to its tool call
    if messages[-1]['role'] == 'assistant' or messages[-1]['role'] == 'system':
        query = input("Enter your message: ")
        if query == "quit":
            break

        message = ChatCompletionUserMessageParam(role='user',
                                                 content=query)
        messages.append(message)

    # This is common gpt call like in tasks 1-3 with added tool options
    response = client_gpt.chat.completions.create(
        model=gpt_model,
        messages=messages,
        tool_choice='auto',
        tools=tools,
    )
    choice = response.choices[0]

    # Gpt can mix tool call with common answer. We save the answer and extract content of the message
    assistant_msg = ChatCompletionAssistantMessageParam(role='assistant',
                                                        content=choice.message.content,
                                                        tool_calls=choice.message.tool_calls)
    messages.append(assistant_msg)
    if choice.message.content is not None:
        print(choice.message.content)

    # if there was at least one tool call in the response, we evaluate it on our side and we append a tool response
    # There has to be tool response for every request before adding any other message
    if choice.message.tool_calls:
        for tool in choice.message.tool_calls:
            idcall = tool.id
            print(f"Calling {tool.function.name}({tool.function.arguments})")
            response = str(eval(tool.function.name)(**json.loads(tool.function.arguments)))
            tool_msg = ChatCompletionToolMessageParam(role='tool',
                                                      tool_call_id=idcall,
                                                      content=response)
            messages.append(tool_msg)



