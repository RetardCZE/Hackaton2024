"""
TASK 3 - Solution
Add chat history as context for the model. Also collect the token usage.
"""

from AI_Tutorial.akkodis_clients import client_gpt_4o

# checkout these objects, ofc you can use dictionaries as presented on openai website
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionUserMessageParam,
                               ChatCompletionSystemMessageParam)

# let's be civil and use typing
from typing import List, Tuple


class Chat:
    def __init__(self, system_msg_content: str) -> None:
        # create an openai client
        self.client, self.model = client_gpt_4o()

        # token usage counter
        self.usage = 0

        # messages list (history)
        self.messages: List[ChatCompletionAssistantMessageParam |
                            ChatCompletionUserMessageParam |
                            ChatCompletionSystemMessageParam] = []

        # initial system message
        system_msg = ChatCompletionSystemMessageParam(role='system', content=system_msg_content)
        self.messages.append(system_msg)

    def send_prompt(self, prompt: str) -> Tuple[str, int]:
        """Send one message with user role and return only text content of the answer and token usage of the request.

        Args:
            prompt: Input from the user.

        Returns:
            Tuple[str, int]
            Content of the response from the language model.
            Total token usage of the request-response (input tokens + generated tokens)
        """

        # prepare user message and append it to history (add it to the conversation)
        user_msg = ChatCompletionUserMessageParam(role='user', content=prompt)
        self.messages.append(user_msg)

        # send completion request for the whole conversation
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )

        # increment total chat usage
        self.usage += response.usage.total_tokens

        # read and save response message (append to the saved conversation)
        assistant_msg = ChatCompletionAssistantMessageParam(role='assistant',
                                                            content=response.choices[0].message.content)
        self.messages.append(assistant_msg)

        # return content of the response and token consumption for the message
        return response.choices[0].message.content, response.usage.total_tokens


if __name__ == "__main__":
    """
    Loop for sending user prompts to llm and printing the response and token usage information.
    
    Study the token consumption. You will see that in subsequent requests their token usage is only growing even if
    both your question and the answer were short. That is caused by fact that you are sending the whole history with
    every request. (Now you see that its good to think about context size as it can quickly grow.)
    """
    chat = Chat("You are a helpful assistant.")
    while 1:
        message = input('-' * 80 + "\nWrite your message: ")
        if message == 'quit':
            break
        response, total_usage = chat.send_prompt(message)
        print(response)
        print(f"Total usage: {chat.usage}")
        print(f"Request usage: {total_usage}")
