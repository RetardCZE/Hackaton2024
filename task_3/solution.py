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
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionUserMessageParam,
                               ChatCompletionSystemMessageParam)
import os
from typing import List, Tuple


class Chat:
    def __init__(self, system_msg_content: str) -> None:
        # create an openai client
        self.client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        self.messages: List[ChatCompletionAssistantMessageParam |
                            ChatCompletionUserMessageParam |
                            ChatCompletionSystemMessageParam] = []
        system_msg = ChatCompletionSystemMessageParam(role='system', content=system_msg_content)
        self.messages.append(system_msg)
        self.usage = 0

    def send_prompt(self, prompt: str) -> Tuple[str, int]:
        user_msg = ChatCompletionUserMessageParam(role='user', content=prompt)
        self.messages.append(user_msg)
        response = self.client.chat.completions.create(
            model="gpt-4o",  # You can use other models as well
            messages=self.messages,
        )
        self.usage += response.usage.total_tokens
        assistant_msg = ChatCompletionAssistantMessageParam(role='assistant',
                                                            content=response.choices[0].message.content)
        self.messages.append(assistant_msg)
        return response.choices[0].message.content, self.usage


if __name__ == "__main__":
    """ loop for sending user prompts to llm and printing the response """
    chat = Chat("You are a helpful assistant.")
    while 1:
        message = input("Write your message: ")
        if message == 'quit':
            break
        response, total_usage = chat.send_prompt(message)
        print(response)
        print(f"Total usage: {total_usage}")
