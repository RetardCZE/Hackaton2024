"""
TASK 8
Now we use chat completion for analysis of the chat

If you modify yor implementation from Task 7 to be reusable, you can use your code,
but I think you can use import from solution 7.

Here you should implement the query loop as in tasks before with the difference, that you want to send the
retrieved context to gpt to answer your query.

So the chain would be:
 - user input -> query
 - retrieval
 - retrievel -> context string (so you can send it as message)
 - query + context -> prompt (make message for the gpt - Given this context: {context} answer this question: {query}.)
      * ofc you should test different prompts to see how important good prompt can be
 - send the promp to gpt as in task 1,2,3

It should work as basic bible bot. But there will still be issues.
For example if you ask conversational context question, the hardcoded retrieval chain will cause some troubles.
(for example:
 'How did Jesus died?' - crucification
 'Who did it to him?' - *Now either there is no history and gpt does not know what 'it' and 'him' represents.
                        * with history you will still get some junk retrieval as in bible a lot of people did something
                        to somebody (you are retrieving now based only on the query)

Think about possible improvements. (Sure bible is quite Jesus-centric, so it may be a bad example)
"""

import numpy as np

from AI_Tutorial.akkodis_clients import client_gpt_4o
from AI_Tutorial.task_7.solution import get_bible_vectorstore

# we reuse task 8 to get index, metadata, the dictionary, embedding client and embedding model
index, metadata, the_bible_dictionary, client_ada, model = get_bible_vectorstore()

# this time we will need client for chat completion
client_gpt, gpt_model = client_gpt_4o()



