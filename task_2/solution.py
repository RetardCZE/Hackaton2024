"""
TASK 2
Make simple one prompt script with user input loop.

do list:
 - Complete the implementation
 - Try some prompts - try something like:
        - What are some good edible fish?
        - Can you give me some recipe?
        Is the answer to the 2nd question fish related?
"""

# import openai for the AI stuff, os for importing the key from environment
import openai
import os

# create an openai client
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))


def send_prompt(prompt: str) -> str:
    """
    Send one message with user role and return only text content of the answer (first choice)
    you can use system message:
    {
        'role': 'system',
        'content': "You are a helpful assistant"
    },
    """
    return client.chat.completions.create(
        model="gpt-4o",  # You can use other models as well
        messages=[
            {
                'role': 'system',
                'content': "You are a helpful assistant"
            },
            {
                'role': 'user',
                'content': f"{prompt}"
            }
        ],
    ).choices[0].message.content


if __name__ == "__main__":
    """ loop for sending user prompts to llm and printing the response """
    while 1:
        message = input("Write your message: ")
        if message == 'quit':
            break
        print(send_prompt(message))

