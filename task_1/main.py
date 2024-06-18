"""
TASK 1
Try out if you can use openAI endpoint.

do list:
 - Run this script. Do not proceed until its working.
 - Inspect response from the endpoint. (most important are choices and usage)
"""

# import openai for the AI stuff, os for importing the key from environment and json for nice print
import openai
import os
import json

# create an openai client which can be used for calling any openai endpoint (not only chat completion)
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))


# Send an example completion prompt
response = client.chat.completions.create(
    model="gpt-4o",  # You can use other models as well
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

# different kinds of output
# inspect style of the response
print(response)
print("ChatCompletion(...) ->")
print(json.dumps(response.dict(), indent=4))
print()
print(response.choices[0].message.content)
