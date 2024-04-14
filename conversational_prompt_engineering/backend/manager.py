import logging
import os

logging.basicConfig(format='%(asctime)s %(message)s')

REQUEST_APIKEY_STRING = "Hello!\nPlease provide your BAM API key with no spaces"


class Manager():
    def __init__(self):
        # init BAM client
        if "BAM_APIKEY" in os.environ:
            self.apikey_set = True
        else:
            self.apikey_set = False
        pass

    def call(self, messages):
        logging.info("conversation so far:")
        if not self.apikey_set:
            if len(messages[-1]['content']) == 47:
                self.apikey_set = True
                # TODO initialize client with the API key
                return "Successfully connected to BAM.\nCan you tell me about your summarization task?"
            else:
                return REQUEST_APIKEY_STRING
        for message in messages:
            print(f"{message['role']}:{message['content']}")
        return "BAM client is not implemented yet"
