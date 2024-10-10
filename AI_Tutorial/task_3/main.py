"""
TASK 3
Add chat history as context for the model. Also collect the token usage.

To have a true conversation experience you have to include conversation history to each prompt
so the completion is relevant not only in context of the last message, but also in context of the whole history.

Implement a 'Chat' object which holds this history (it doesn't have to be class, but why not since we are in python).

When you implement all the code, see how token usage behaves. You may find interesting that even short messages
later in the conversation consume a lot of tokens (that is because you send the whole conversation every time).
"""

from AI_Tutorial.akkodis_clients import client_gpt_4o

# checkout these objects, ofc you can use dictionaries as presented on openai website
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionUserMessageParam,
                               ChatCompletionSystemMessageParam)

# let's be civil and use typing
from typing import List, Tuple

'''
********************************************************************************
TODO: implement chat class for chatting with conversation context included
'''
class Chat:
    def __init__(self, system_msg_content: str) -> None:
        self.client, self.model = client_gpt_4o()

        # define container for the history and counter for token usage
        self.usage = 0
        # self.messages

        # and add system message as first message

    def send_prompt(self, prompt: str) -> Tuple[str, int]:
        """Send one message with user role and return only text content of the answer and token usage of the request.
        """

        # prepare user message and append it to history (add it to the conversation)

        # send completion request for the whole conversation
        # response =

        # increment total chat usage

        # read and save response message (append to the saved conversation)
        # for example
        # assistant_msg = ChatCompletionAssistantMessageParam(role='assistant',
        #                                                     content=...)
        # self.messages <- assistant_msg

        # return content of the response and token consumption for the message
        return "", 0


if __name__ == "__main__":
    """
    Loop for sending user prompts to llm and printing the response and token usage information.
    Study the token consumption.
    """
    chat = Chat(system_msg_content="You are a helpful assistant.")
    while 1:
        message = input('-' * 80 + "\nWrite your message: ")
        if message == 'quit':
            break
        response, total_usage = chat.send_prompt(message)
        print(response)
        print(f"Total usage: {chat.usage}")
        print(f"Request usage: {total_usage}")
