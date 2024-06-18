"""
TASK 3
Add chat history as context for the bot also collect the token usage.
And let's do the chat as an object this time.

do list:
    - implement the Chat class
    - try out questions from task 2. What changed?
"""

# import openai for the AI stuff, os for importing the key from environment
import openai
# checkout these objects, ofc you can use dictionaries as presented on openai website
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionUserMessageParam,
                               ChatCompletionSystemMessageParam)
import os
# let's be civil and use typing
from typing import List, Tuple


class Chat:
    def __init__(self, system_msg_content: str) -> None:
        # create an openai client
        self.client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        # now you need to prepare something for saving messages and usage.. and dont forget to add system message
        # self.messages
        self.usage = 0

    def send_prompt(self, prompt: str) -> Tuple[str, int]:
        """
        Here you want to prepare messages for completion endpoint. You have to send the whole conversation every time.
        Don't forget to save both user and assistant message.
        """
        return "Text of the response", self.usage


if __name__ == "__main__":
    chat = Chat("You are a helpful assistant.")
    while 1:
        message = input("Write your message: ")
        if message == 'quit':
            break
        response, total_usage = chat.send_prompt(message)
        print(response)
        print(f"Total usage: {total_usage}")