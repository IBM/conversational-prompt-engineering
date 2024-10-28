# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
from urllib.parse import quote_plus
import sys

# from dotenv import load_dotenv


from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    TextGenerationParameters, )

from conversational_prompt_engineering.backend.util.llm_clients.abst_llm_client import HumanRole
from conversational_prompt_engineering.backend.util.llm_clients.abst_watson_client import AbstWatsonClient





class BamClient(AbstWatsonClient):
    def __init__(self, api_endpoint, model_params):
        super(BamClient, self).__init__(model_id=model_params['watsonx_model_id'])
        self.client = Client(credentials=Credentials(api_key=self._get_env_var('BAM_APIKEY'), api_endpoint=api_endpoint))
        self.parameters = model_params

    @classmethod
    def display_name(self):
        return "Bam"

    @classmethod
    def credentials_params(cls):
        return {"BAM_APIKEY": "BAM API key"}

    def prompt_llm(self, conversation, max_new_tokens=None):
        parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.GREEDY,
            max_new_tokens=max_new_tokens if max_new_tokens else self.parameters['max_new_tokens'],
            min_new_tokens=1,
            repetition_penalty=self.parameters['repetition_penalty'] if 'repetition_penalty' in self.parameters else 1
            )
        response = self.client.text.generation.create(
            model_id=self.model_id,
            inputs=[conversation],
            parameters=parameters,
        )
        texts = [res.generated_text.strip() for resp in response for res in resp.results]
        return texts
