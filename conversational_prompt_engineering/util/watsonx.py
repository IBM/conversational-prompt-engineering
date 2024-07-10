import logging
import sys


from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference

from conversational_prompt_engineering.util.abst_generate import AbstGenerate



class WatsonXGenerate(AbstGenerate):
    def __init__(self, params):
        super(WatsonXGenerate, self).__init__()
        self.parameters = params
        self.api_endpoint = params["api_endpoint"]
        self.project_id = params["project_id"]
        self.api_key = params["api_key"]

        credentials = {
            "url": self.api_endpoint,
            "apikey": self.api_key
        }
        self.client = APIClient(credentials)
        self.client.set.default_project(self.project_id)

        self.generate_params = {
                    GenParams.MAX_NEW_TOKENS: self.parameters['max_new_tokens'],
                    GenParams.DECODING_METHOD: 'greedy',
                    GenParams.MIN_NEW_TOKENS: 1,
                    GenParams.TRUNCATE_INPUT_TOKENS: self.parameters['max_total_tokens'],
                    GenParams.REPETITION_PENALTY : self.parameters['repetition_penalty'] if 'repetition_penalty' in self.parameters else 1
                }
        self.model_id =  self.parameters['model_id']


    def _get_moded(self, max_new_tokens=None):
        params = {x: y for x, y in self.generate_params.items()}
        if max_new_tokens:
            params[GenParams.MAX_NEW_TOKENS] = max_new_tokens
        return ModelInference(
                    model_id=self.model_id,
                    params=self.generate_params,
                    api_client=self.client
            )


    def do_send_messages(self, conversation, max_new_tokens=None):
        sys.tracebacklimit = 1000
        model = self._get_moded(max_new_tokens)
        for i in [0,1]:
            try:
                res = model.generate_text(prompt=[conversation])
                texts = [x.strip() for x in res]
                return texts
            except Exception as e:
                if i == 0:
                    logging.warning(
                        f"ERROR Got API response exception: {e.response.model_dump_json()}")
                else:
                    logging.error("ERROR Got API response exception", e)
        sys.tracebacklimit = 0
        raise Exception("There is an error connecting to WatsonX. Either check your API key or try again in a few minutes.")

