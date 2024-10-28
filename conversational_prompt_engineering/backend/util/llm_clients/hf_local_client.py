# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import ModelInference
from huggingface_hub import InferenceClient
import transformers

from conversational_prompt_engineering.backend.util.llm_clients.abst_hf_client import AbstHFClient

class HFLocalClient(AbstHFClient):

    @classmethod
    def credentials_params(cls):
        return {}

    @classmethod
    def display_name(self):
        return "HF Endpoint"

    def __init__(self, api_endpoint, model_params):
        super(HFLocalClient, self).__init__(model_id=model_params["hf_local_model_id"])
        self.parameters = model_params

        import torch

        #default generation strategy is greedy
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",

        )


    def prompt_llm(self, conversation, max_new_tokens=None):
        if not max_new_tokens:
            max_new_tokens = self.parameters['max_new_tokens']

        outputs = self.pipeline(text_inputs=conversation,
                           max_new_tokens=max_new_tokens,
                           max_length=self.parameters['max_total_tokens'],
                            pad_token_id=self.pipeline.tokenizer.eos_token_id)
        return [outputs[0]["generated_text"][-1]['content']]

