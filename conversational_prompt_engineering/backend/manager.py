import json
import logging
import os
from enum import Enum

from conversational_prompt_engineering.util.bam import BAMChat, HumanRole

logging.basicConfig(format='%(asctime)s %(message)s')

REQUEST_APIKEY_STRING = "Hello!\nPlease provide your BAM API key with no spaces"
OK_OR_CHANGE = "After that, ask User if the summary is ok for them, or would they like to change anything. " \
               "Wait for their response. Based on their response, update the instruction. Continue this process until User has no additional feedback. Write 'I understand that the summary is ok.'"


class DialogState(Enum):
    PredefinedQuestions = 1
    ExampleDrivenPromptUpdate = 2
    FinalInstruction = 3


class Manager():
    def __init__(self):
        self.dialog_state = DialogState.PredefinedQuestions
        self.admin_params = self.load_admin_params()
        if "BAM_APIKEY" in os.environ:
            self.apikey_set = True
            params = self.load_bam_params()
            self.bam_client = BAMChat(params, self.admin_params['stage_1']['prompt'])
        else:
            self.apikey_set = False  # TODO: handle this flow
            self.bam_client = None

    def load_admin_params(self):
        with open("backend/admin_params.json", "r") as f:
            params = json.load(f)
        return params

    def load_bam_params(self):
        with open("backend/bam_params.json", "r") as f:
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
                return "Successfully connected to BAM.\nHi"
            else:
                return REQUEST_APIKEY_STRING
        user_message = messages[-1]
        response = self.bam_client.send_message(user_message['content'], HumanRole.User)
        response = self.interfer_if_needed(response)
        return response

    def interfer_if_needed(self, response_to_user):
        include_admin_response = True
        response_to_admin = ""
        if self.admin_params['stage_1']['finish_signal'] in response_to_user and self.dialog_state == DialogState.PredefinedQuestions:
            response_to_admin = self.bam_client.send_message(self.admin_params['stage_2']['prompt'], HumanRole.Admin)
            self.dialog_state = DialogState.ExampleDrivenPromptUpdate
            return response_to_user + "\n\n[RESPONSE TO ADMIN]" + response_to_admin
        elif self.admin_params['stage_2']['finish_signal'] in response_to_user and self.dialog_state == DialogState.ExampleDrivenPromptUpdate:
            response_to_admin = self.bam_client.send_message(self.admin_params['stage_3']['prompt'], HumanRole.Admin)
            self.dialog_state = DialogState.FinalInstruction
            return response_to_user + "\n\n[RESPONSE TO ADMIN]" + response_to_admin
        else:
            return response_to_user
        # if include_admin_response:
        #     return response_to_user + "\n\n" + response_to_admin
        # else:
        #     return response_to_user
