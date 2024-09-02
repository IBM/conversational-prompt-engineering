from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference

from conversational_prompt_engineering.backend.util.llm_clients.abst_llm_client import AbstLLMClient



class WatsonXClient(AbstLLMClient):

    @classmethod
    def credentials_params(cls):
        return {"WATSONX_APIKEY": "Watsonx API key",
                        "PROJECT_ID": "project_id"}

    @classmethod
    def display_name(self):
        return "WatsonX"

    def __init__(self, api_endpoint, model_params):
        super(WatsonXClient, self).__init__()
        self.parameters = model_params
        self.api_endpoint = api_endpoint
        self.project_id = self._get_env_var("PROJECT_ID")
        self.api_key = self._get_env_var("WATSONX_APIKEY")

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

    def _get_model(self, max_new_tokens=None):
        params = {x: y for x, y in self.generate_params.items()}
        if max_new_tokens:
            params[GenParams.MAX_NEW_TOKENS] = max_new_tokens
        return ModelInference(
                    model_id=self.model_id,
                    params=self.generate_params,
                    api_client=self.client
            )

    def prompt_llm(self, conversation, max_new_tokens=None):
        model = self._get_model(max_new_tokens)
        res = model.generate_text(prompt=[conversation])
        texts = [x.strip() for x in res]
        return texts

