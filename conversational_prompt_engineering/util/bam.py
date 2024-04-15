import os
from enum import Enum

import pandas as pd
#from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    SystemMessage,
    TextGenerationParameters,
)
from tqdm import tqdm

# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>

class HumanRole(Enum):
    User="user"
    Admin="admin"

class BAMChat:
    def __init__(self, params, system_prompt=""):
        #load_dotenv()
        self.client = Client(credentials=Credentials(api_key=params['api_key'], api_endpoint=params['api_endpoint']))
        self.parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.GREEDY, max_new_tokens=500, min_new_tokens=1
        )
        self.model_id = params['model_id']
        self.system_prompt = system_prompt
        self.conversation_id = None

    def send_message(self, text, message_human_role:HumanRole):
        text = f'{message_human_role.name}: {text}'
        messages = [HumanMessage(content=text)]
        if self.conversation_id is None:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.client.text.chat.create(
            conversation_id=self.conversation_id,
            model_id=self.model_id,
            messages=messages,
            parameters=self.parameters,
        )
        self.conversation_id = response.conversation_id
        return response.results[0].generated_text # TODO: return conversation_id ?

    # TODO: inference for few-shot examples should use generate?
    def bam_infer(self):
        pass