# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference
from huggingface_hub import InferenceClient

from conversational_prompt_engineering.backend.util.llm_clients.abst_hf_client import AbstHFClient

class HFRemoteClient(AbstHFClient):

    @classmethod
    def credentials_params(cls):
        return {"HF_TOKEN": "Huggingface Token"}

    @classmethod
    def display_name(self):
        return "HF Endpoint"

    def __init__(self, api_endpoint, model_params):
        super(HFRemoteClient, self).__init__(model_id=model_params['hf_model_id'])
        self.parameters = model_params
        self.hf_token = self._get_env_var("HF_TOKEN")


    def prompt_llm(self, conversation, max_new_tokens=None):
        if not max_new_tokens:
            max_new_tokens = self.parameters['max_new_tokens']
        client = InferenceClient(
            self.model_id,
            token=self.hf_token,
        )
        try:
            #default generation strategy is greedy
            messages = client.chat_completion(
                messages=conversation,
                max_tokens=max_new_tokens,
                stream=False,
            )
        except Exception as e:
            raise Exception(e.response.content)
        res = messages.choices[0].message.content
        return [res]

    """
     
     :param conversation: 
     :param max_new_tokens: 
     :return: 
     


        return messages[0]

    """
