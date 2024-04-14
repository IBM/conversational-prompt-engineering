import json
import logging
import os

from conversational_prompt_engineering.util.bam import BAMChat

logging.basicConfig(format='%(asctime)s %(message)s')

REQUEST_APIKEY_STRING = "Hello!\nPlease provide your BAM API key with no spaces"


class Manager():
    def __init__(self):
        if "BAM_APIKEY" in os.environ:
            self.apikey_set = True
            params = self.load_bam_params()
            self.bam_client = BAMChat(params)
        else:
            self.apikey_set = False # TODO: handle this flow
            self.bam_client = None

    def load_bam_params(self):
        with open("backend/params.json", "r") as f:
            params = json.load(f)
        params['api_key'] = os.getenv("BAM_APIKEY")
        return params

    def call(self, messages):
        logging.info("conversation so far:")
        if not self.apikey_set:
            if len(messages[-1]['content']) == 47:
                self.apikey_set = True
                params = self.load_bam_params()
                params['api_key'] = messages[-1]['content']
                self.bam_client = BAMChat(params)
                return "Successfully connected to BAM.\nCan you tell me about your summarization task?"
            else:
                return REQUEST_APIKEY_STRING
        user_message = messages[-1]
        response = self.bam_client.send_message(user_message['content'])
        return response

