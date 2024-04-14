import json
import logging
import os
from enum import Enum

from conversational_prompt_engineering.util.bam import BAMChat, HumanRole

logging.basicConfig(format='%(asctime)s %(message)s')

REQUEST_APIKEY_STRING = "Hello!\nPlease provide your BAM API key with no spaces"
OK_OR_CHANGE = "After that, ask USER if the summary is ok for them, or would they like to change anything. " \
               "Wait for their response. Based on their response, update the instruction, and share it with them with the prefix 'updated instruction:'"


class DialogState(Enum):
    Intro = 1
    InstructionReceived = 2
    GenerateSummaryReceiveFeedback = 3
    UpdateInstruction = 4


class Manager():
    def __init__(self):
        self.dialog_state = DialogState.Intro
        if "BAM_APIKEY" in os.environ:
            self.apikey_set = True
            params = self.load_bam_params()
            self.bam_client = BAMChat(params)
        else:
            self.apikey_set = False  # TODO: handle this flow
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
        response = self.bam_client.send_message(user_message['content'], HumanRole.User)
        response = self.interfer_if_needed(response)
        return response

    def interfer_if_needed(self, response_to_user):
        include_admin_response = True
        if "instruction:" in response_to_user.lower() and self.dialog_state == DialogState.Intro:
            response_to_admin = self.bam_client.send_message(
                "Thanks. Now ask USER to send you a text that you will summarize with this instruction. Do not provide any other information.\n" +
                OK_OR_CHANGE,
                HumanRole.Admin)
            self.dialog_state = DialogState.InstructionReceived
        elif self.dialog_state == DialogState.InstructionReceived:
            response_to_admin = self.bam_client.send_message("Is the user satisfied with the summary?\nAnswer ONLY yes or no without any additional text.", HumanRole.Admin)
            if response_to_admin.lower()=='yes':
                response_to_admin = self.bam_client.send_message(
                    "Thanks. Now combine the instruction, text and summary to share with the USER a prompt that they can use to fully utilize the power of your model in text summarization.",
                    HumanRole.Admin)
            else:
                include_admin_response = False
        else:
            return response_to_user
        if include_admin_response:
            return response_to_user + "\n" + response_to_admin
        else:
            return response_to_user
