import logging
from urllib.parse import quote_plus
import sys

# from dotenv import load_dotenv


from genai.client import Client
from genai.credentials import Credentials
from genai.exceptions import ApiResponseException, ApiNetworkException, ValidationError
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    SystemMessage,
    TextGenerationParameters, )
from tqdm import tqdm

from conversational_prompt_engineering.backend.util.llm_clients.abst_llm_client import AbstLLMClient, HumanRole


# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>




class BamClient(AbstLLMClient):
    def __init__(self, credentials, api_endpoint, model_params):
        super(BamClient, self).__init__()
        self.client = Client(credentials=Credentials(api_key=credentials['BAM_APIKEY'], api_endpoint=api_endpoint))
        self.parameters = model_params

    @classmethod
    def display_name(self):
        return "Bam"

    @classmethod
    def credentials_params(cls):
        return {"BAM_APIKEY": "BAM API key"}

    def do_send_messages(self, conversation, max_new_tokens=None):
        sys.tracebacklimit = 1000
        for i in [0,1]:
            try:
                parameters = TextGenerationParameters(
                    decoding_method=DecodingMethod.GREEDY,
                    max_new_tokens=max_new_tokens if max_new_tokens else self.parameters['max_new_tokens'],
                    min_new_tokens=1,
                    repetition_penalty=self.parameters['repetition_penalty'] if 'repetition_penalty' in self.parameters else 1
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
        raise Exception("There is an error connecting to BAM. Either check your API key or try again in a few minutes.")
