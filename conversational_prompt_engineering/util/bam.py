import os
from enum import Enum
from urllib.parse import quote_plus

import pandas as pd
# from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    SystemMessage,
    TextGenerationParameters, ChatRole,
)
from tqdm import tqdm


# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>

class HumanRole(Enum):
    User = "user"
    Admin = "admin"


class BAMChat:
    def __init__(self, params, system_prompt=""):
        #load_dotenv()
        self.client = Client(credentials=Credentials(api_key=params['api_key'], api_endpoint=params['api_endpoint']))
        self.params = params
        self.system_prompt = system_prompt
        self.conversation_id = None

    def send_message(self, text, message_human_role:HumanRole, override_params=None):
        if override_params is None:
            override_params = {}
        text = f'{message_human_role.name}: {text}'
        messages = [HumanMessage(content=text)]
        if self.conversation_id is None:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        params = {**self.params, **override_params}
        text_generation_parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.GREEDY, max_new_tokens=params['max_new_tokens']
        )
        response = self.client.text.chat.create(
            conversation_id=self.conversation_id,
            model_id=params['model_id'],
            messages=messages,
            parameters=text_generation_parameters,
        )
        self.conversation_id = response.conversation_id
        return response.results[0].generated_text  # TODO: return conversation_id ?

    # TODO: inference for few-shot examples should use generate?
    def bam_infer(self):
        pass


class BamGenerate:
    def __init__(self, params):
        self.client = Client(credentials=Credentials(api_key=params['api_key'], api_endpoint=params['api_endpoint']))
        self.parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.GREEDY, max_new_tokens=params['max_new_tokens'], min_new_tokens=1
        )
        self.model_id = params['model_id']

    def send_messages(self, conversation):
        response = self.client.text.generation.create(
            model_id=self.model_id,
            inputs=[conversation],
            parameters=self.parameters,
        )
        texts = [res.generated_text.strip() for resp in response for res in resp.results]
        return texts

    def save_prompt(self, name, text):
        count = 0
        res_name = name
        while res_name in [found.name for found in self.client.prompt.list(search=name).results]:
            count += 1
            res_name = f'{name}_{count}'

        self.client.prompt.create(name=res_name, model_id=self.model_id, input=text, task_id='summarization')
        link = f'https://bam.res.ibm.com/lab?model={quote_plus(self.model_id)}&mode=freeform'
        return res_name, link
