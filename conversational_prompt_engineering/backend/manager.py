import json
import logging
import os
from enum import Enum

from conversational_prompt_engineering.util.bam import BAMChat, HumanRole

logging.basicConfig(format='%(asctime)s %(message)s')

OK_OR_CHANGE = "After that, ask User if the summary is ok for them, or would they like to change anything. " \
               "Wait for their response. Based on their response, update the instruction. Continue this process until User has no additional feedback. Write 'I understand that the summary is ok.'"


class DialogState(Enum):
    PredefinedQuestions = 1
    ExampleDrivenPromptUpdate = 2
    SummarizeExample = 3
    FinalInstruction = 4


class Mode(Enum):
    Basic = 1
    Advanced = 2


def build_final_prompt(response_to_admin):
    try:
        prompt_json = json.loads(response_to_admin)
        prompt = prompt_json['instruction'] + "\n\n" + "\n\n".join([f'Text: {t}\n\nSummary: {s}' for t, s in
                                                                    zip(prompt_json['texts'], prompt_json[
                                                                        'summaries'])]) + "\n\nText: {your_text}\n\nSummary: "
    except:
        prompt = f"Something went wrong with building the final instruction:\n\n{response_to_admin}"
    return prompt


class Manager():
    def __init__(self, mode, bam_api_key):
        self.dialog_state = DialogState.PredefinedQuestions
        self.admin_params = self.load_admin_params()

        self.apikey_set = True
        params = self.load_bam_params()
        params['api_key'] = bam_api_key
        self.bam_client = BAMChat(params, self.admin_params['stage_1']['prompt'])
        self.mode = mode

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
        for message in messages:
            logging.info(f"{message['role']}:{message['content']}")
        user_message = messages[-1]
        response = self.bam_client.send_message(user_message['content'], HumanRole.User)
        if self.mode == Mode.Basic:
            response = self.basic_interfere_if_needed(response)
        else:
            response = self.interfere_if_needed(response)
        return response

    # # # From stage 1 to 2
    def interfere_if_needed(self, response_to_user):
        include_admin_response = True
        response_to_admin = ""
        if self.admin_params['stage_1'][
            'finish_signal'] in response_to_user.lower() and self.dialog_state == DialogState.PredefinedQuestions:
            response_to_admin = self.bam_client.send_message(self.admin_params['stage_2']['prompt'], HumanRole.Admin)
            self.dialog_state = DialogState.ExampleDrivenPromptUpdate
            return response_to_user + "\n\n[RESPONSE TO ADMIN]" + response_to_admin
        elif self.admin_params['stage_2'][
            'finish_signal'] in response_to_user and self.dialog_state == DialogState.ExampleDrivenPromptUpdate:
            response_to_admin = self.bam_client.send_message(self.admin_params['stage_3']['prompt'], HumanRole.Admin)
            self.dialog_state = DialogState.SummarizeExample
            return response_to_user + "\n\n[RESPONSE TO ADMIN]" + response_to_admin
        elif self.admin_params['stage_3'][
            'finish_signal'] in response_to_user.lower() and self.dialog_state == DialogState.SummarizeExample:
            response_to_admin = self.bam_client.send_message(self.admin_params['stage_4']['prompt'], HumanRole.Admin)
            self.dialog_state = DialogState.FinalInstruction
            prompt = build_final_prompt(response_to_admin)
            return response_to_user + "\n\n[RESPONSE TO ADMIN] Here is the final few-shot prompt:\n\n" + prompt
        else:
            return response_to_user

    # From stage 1 to stage 3
    def basic_interfere_if_needed(self, response_to_user):
        include_admin_response = True
        response_to_admin = ""
        if self.admin_params['stage_1'][
            'finish_signal'] in response_to_user.lower() and self.dialog_state == DialogState.PredefinedQuestions:
            response_to_admin = self.bam_client.send_message(self.admin_params['stage_3']['prompt'],
                                                             HumanRole.Admin)  ### changed from 2
            self.dialog_state = DialogState.SummarizeExample
            return response_to_user + "\n\n[RESPONSE TO ADMIN]" + response_to_admin
        elif self.admin_params['stage_3'][
            'finish_signal'] in response_to_user.lower() and self.dialog_state == DialogState.SummarizeExample:
            response_to_admin = self.bam_client.send_message(self.admin_params['stage_4']['prompt'], HumanRole.Admin)
            self.dialog_state = DialogState.FinalInstruction
            prompt = build_final_prompt(response_to_admin)
            return response_to_user + "\n\n[RESPONSE TO ADMIN] Here is the final few-shot prompt:\n\n" + prompt
        else:
            return response_to_user
