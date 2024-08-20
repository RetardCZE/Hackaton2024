"""
TASK 2 - Solution
Make a simple script to resolve user questions.
"""

# import and initialise the client
from AI_Tutorial.akkodis_clients import client_gpt_4o
client, model = client_gpt_4o()

def send_prompt(prompt: str) -> str:
    """Send one message with user role and return only text content of the answer.

    Args:
        prompt: Input from the user.

    Returns:
        Content of the response from the language model.
    """
    return client.chat.completions.create(
        model=model,  # You can use other models as well
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

