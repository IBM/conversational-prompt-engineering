import json
import logging
import os
from enum import Enum

from conversational_prompt_engineering.util.bam import BAMChat, HumanRole

app_logger = logging.getLogger()
app_logger.addHandler(logging.StreamHandler())
app_logger.setLevel(logging.INFO)

OK_OR_CHANGE = "After that, ask User if the summary is ok for them, or would they like to change anything. " \
               "Wait for their response. Based on their response, update the instruction. Continue this process until User has no additional feedback. Write 'I understand that the summary is ok.'"


class DialogState(Enum):
    PredefinedQuestions = "stage_1"
    ExampleDrivenPromptUpdate = "stage_2"
    ExampleDrivenPromptUpdate1 = "stage_2_1"
    ExampleDrivenPromptUpdate2 = "stage_2_2"
    SummarizeExample = "stage_3"
    FinalInstruction = "stage_4"
    # EditSummaries = "stage_5"



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
        self.bam_client = BAMChat(params, self.admin_params[self.dialog_state.value]['prompt'])
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
        response = self.interfere_if_needed(response)
        logging.info(f"Stage {self.dialog_state}")
        return response

    def interference_condition(self, response_to_user):
        logging.info(f"trying interference in stage {self.dialog_state}")
        stage_str = self.dialog_state.value
        if stage_str == "stage_2_2":  # manual signal
            return self.admin_params[stage_str]['finish_signal'] in response_to_user.lower()
        if stage_str != 'stage_4':
            response_to_admin = self.bam_client.send_message(self.admin_params[stage_str]['interference'],
                                                             HumanRole.Admin, override_params={'max_new_tokens': 1})
        else:
            response_to_admin = "no"
        logging.info(f"response to interference in stage {self.dialog_state}: {response_to_admin}")
        return response_to_admin.lower().strip().startswith("yes")

    def get_next_stage(self):
        if self.dialog_state == DialogState.PredefinedQuestions:
            return DialogState.ExampleDrivenPromptUpdate if self.mode == Mode.Advanced else DialogState.SummarizeExample
        elif self.dialog_state == DialogState.ExampleDrivenPromptUpdate:
            return DialogState.ExampleDrivenPromptUpdate1
        elif self.dialog_state == DialogState.ExampleDrivenPromptUpdate1:
            return DialogState.ExampleDrivenPromptUpdate2
        elif self.dialog_state == DialogState.ExampleDrivenPromptUpdate2:
            return DialogState.SummarizeExample
        elif self.dialog_state == DialogState.SummarizeExample:
            return DialogState.FinalInstruction
        else:
            return None

    def generate_response_to_user(self, response_to_user, response_to_admin):
        if self.dialog_state != DialogState.FinalInstruction:
            return response_to_user + "\n\n[RESPONSE TO ADMIN]" + response_to_admin
        prompt = build_final_prompt(response_to_admin)
        return response_to_user + "\n\n[RESPONSE TO ADMIN] Here is the final few-shot prompt:\n\n" + prompt

    def interfere_if_needed(self, response_to_user):
        if self.interference_condition(response_to_user):
            self.dialog_state = self.get_next_stage()
            response_to_admin = self.bam_client.send_message(self.admin_params[self.dialog_state.value]['prompt'],
                                                             HumanRole.Admin)
            response_to_user = self.generate_response_to_user(response_to_user, response_to_admin)
            return response_to_user
        else:
            return response_to_user


