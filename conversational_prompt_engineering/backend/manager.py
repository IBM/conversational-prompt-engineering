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
        if "i have all the necessary information to build an initial instruction for your text summarization task" in response_to_user.lower() and self.dialog_state == DialogState.Intro:
            response_to_admin = self.bam_client.send_message(
                "Now ask User to send you a text that you will summarize with this instruction. I will pass this question to User. Do not provide any other information.\n" +
                OK_OR_CHANGE,
                HumanRole.Admin)
            self.dialog_state = DialogState.InstructionReceived
        elif "i understand that the summary is ok." in response_to_user.lower() and self.dialog_state == DialogState.InstructionReceived:
            response_to_admin = self.bam_client.send_message(
                "Now combine the updated instruction, the text and the summary to share with User a prompt that they can use to fully utilize the power of your model in text summarization. I will pass the prompt to User.",
                HumanRole.Admin)
            self.dialog_state = DialogState.GenerateSummaryReceiveFeedback
        else:
            return response_to_user
        if include_admin_response:
            return response_to_user + "\n\n" + response_to_admin
        else:
            return response_to_user
