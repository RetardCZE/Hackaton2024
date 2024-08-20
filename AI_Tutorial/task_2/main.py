"""
TASK 2
Make a simple script to resolve user questions.

Use client to send input user message to gpt and print the answer.
You can use code from Task 1 and a very simple input loop.
Then try a conversation context question to see the issue with this implementation.
Q1: Give me a list of 5 good edible fish.
Q2: Can you give me recipe for the first one?
    - you will definitely get some random recipe
"""

# import and initialise the client
from AI_Tutorial.akkodis_clients import client_gpt_4o
client, model = client_gpt_4o()

'''
********************************************************************************
TODO: implement a function to send user message to the API chat completion and return response content
'''
def send_prompt(prompt: str) -> str:
    # define messages for the request
    system_message = {'role': 'system',
                      'content': '...'}
    user_message = {'role': 'user',
                    'content': '...'}

    # send chat completions request through the client (messages = [system_message, user_message])
    # response = client.?

    # return content of chat completion response
    return ""


# implement the main loop for user input (input(), send_prompt(), print() and some quit condition is enough)
if __name__ == "__main__":
    while 1:
        pass
