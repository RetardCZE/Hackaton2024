"""
TASK 1
Test if your AKKODIS api key works.

If you added your api key to akkodis_clients.py, this code should be executable.
Run it and see all the prints. Look at the end of the code for description of the prints.
"""

# We use json package for nice print of the response.
import json

# Import client initialisation method.
from AI_Tutorial.akkodis_clients import client_gpt_4o

# Create client and relevant model name.
client, model = client_gpt_4o()

# Use client to send a chat completion request to model gpt-4o.
# You can see that messages has to be sent as list of dictionaries, with 'role' and 'content' keys.
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            'role': 'system',
            'content': "You are poetic assistant trying to make some fun."
        },
        {
            'role': 'user',
            'content': "Tell me something nice about start of our AI learning hackaton. Kind of Hello, hackaton! stuff."
        }
    ],
)

# Print out the response in multiple formats

# print the raw object
print(response)

# nice print in json format
print("ChatCompletion(...) ->")
print(json.dumps(response.model_dump(), indent=4))
print('\n' + '-' * 80 + '\n')

# print only content of the message (see that I had to specify the choice even there is only one available)
print(response.choices[0].message.content)
