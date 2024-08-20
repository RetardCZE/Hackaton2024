""" Tutorial python interface to some AKKODIS api endpoints
@author: Jakub Rosol, jakub.rosol@akkodis.com
@date: 19.08.2024

Do not copy it for productive code.
(Make class where you define the client and add method for usage of needed endpoints)
"""

import openai
from typing import Tuple
from AI_Tutorial.conf import AKKODIS_API_KEY, OPENAI_API_KEY, PROVIDER

API_BASE = "https://cld.akkodis.com/api/openai/"
API_VERSION = "2024-05-01-preview"

def client_gpt_4o() -> Tuple[openai.AzureOpenAI, str]:
    if PROVIDER == "AKKODIS":
        deployment = "models-gpt-4o"
        client = openai.AzureOpenAI(base_url=API_BASE,
                                    api_key=AKKODIS_API_KEY,
                                    api_version=API_VERSION)
    elif PROVIDER == 'OPENAI':
        deployment = "gpt-4o"
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

    return client, deployment

def client_ada_002() -> Tuple[openai.AzureOpenAI, str]:
    if PROVIDER == "AKKODIS":
        deployment = "models-ada-002"
        client = openai.AzureOpenAI(base_url=API_BASE,
                                    api_key=AKKODIS_API_KEY,
                                    api_version=API_VERSION)
    elif PROVIDER == 'OPENAI':
        deployment = "text-embedding-ada-002"
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

    return client, deployment




