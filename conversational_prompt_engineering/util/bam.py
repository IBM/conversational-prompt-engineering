import logging
import os
from enum import Enum
from urllib.parse import quote_plus
import sys

import pandas as pd
# from dotenv import load_dotenv

from genai.client import Client
from genai.credentials import Credentials
from genai.exceptions import ApiResponseException, ApiNetworkException, ValidationError
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
        self.parameters = params

    def count_tokens(self, txt):
        if type(txt) is str:
            txt = [txt]
        token_counts = []
        try:
            for response in tqdm(self.client.text.tokenization.create(
                model_id=self.parameters['model_id'],
                input=txt),
            ):
                for result in response.results:
                    token_counts.append(result.token_count)
            return token_counts
        except ApiResponseException as e:
            logging.warning(f"ERROR Got API response exception: {e.response.model_dump_json()}")  # our handcrafted message
        except ApiNetworkException as e:
            logging.warning(f"ERROR The server could not be reached: {e}")
        except ValidationError as e:
            logging.warning(f"ERROR Provided parameters are not valid: {e}")
        finally:
            return [len(t.split()) for t in txt]  # tokenization failed, fallback to text split

    def send_messages(self, conversation, max_new_tokens=None):
        sys.tracebacklimit = 1000
        for i in [0,1]:
            try:
                parameters = TextGenerationParameters(
                    decoding_method=DecodingMethod.GREEDY,
                    max_new_tokens=max_new_tokens if max_new_tokens else self.parameters['max_new_tokens'],
                    min_new_tokens=1
                )
                response = self.client.text.generation.create(
                    model_id=self.parameters['model_id'],
                    inputs=[conversation],
                    parameters=parameters,
                )
                texts = [res.generated_text.strip() for resp in response for res in resp.results]
                return texts
            except ApiResponseException as e:
                if i == 0:
                    logging.warning(
                        f"ERROR Got API response exception: {e.response.model_dump_json()}")
                else:
                    logging.error("ERROR Got API response exception", e)
            except ApiNetworkException as e:
                if i == 0:
                    logging.warning(f"ERROR The server could not be reached: {e}")
                else:
                    logging.error("ERROR The server could not be reached", e)
            except ValidationError as e:
                if i == 0:
                    logging.warning(f"ERROR Provided parameters are not valid: {e}")
                else:
                    logging.error("ERROR Provided parameters are not valid", e)
        sys.tracebacklimit = 0
        raise Exception("There is an error in BAM. Please try again in a few minutes.")

    def save_prompt(self, name, text):
        count = 0
        res_name = name
        while res_name in [found.name for found in self.client.prompt.list(search=name).results]:
            count += 1
            res_name = f'{name}_{count}'
        model_id = self.parameters['model_id']
        self.client.prompt.create(name=res_name, model_id=model_id, input=text, task_id='summarization')
        link = f'https://bam.res.ibm.com/lab?model={quote_plus(model_id)}&mode=freeform'
        return res_name, link
