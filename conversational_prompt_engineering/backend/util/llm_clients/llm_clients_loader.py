import os
from glob import glob
from conversational_prompt_engineering.backend.util.llm_clients.watsonx_client import WatsonXClient


def get_client_classes(llm_clients_list):
    all_models = [WatsonXClient]
    name_to_models = {x.__name__: x for x in all_models}
    return [name_to_models[x] for x in llm_clients_list]