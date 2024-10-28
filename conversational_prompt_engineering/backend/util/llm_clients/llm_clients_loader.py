# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import os
from glob import glob
from conversational_prompt_engineering.backend.util.llm_clients.bam_client import BamClient
from conversational_prompt_engineering.backend.util.llm_clients.watsonx_client import WatsonXClient
from conversational_prompt_engineering.backend.util.llm_clients.hf_remote_client import HFRemoteClient
from conversational_prompt_engineering.backend.util.llm_clients.hf_local_client import HFLocalClient



def get_client_classes(llm_client):
    all_models = [BamClient, WatsonXClient, HFRemoteClient, HFLocalClient]
    name_to_models = {x.__name__: x for x in all_models}
    client = name_to_models.get(llm_client)
    if client is None:
        raise ValueError(f"Wrong llm_api values {llm_client}. You may only use on of {[x for x in name_to_models]}")
    return client