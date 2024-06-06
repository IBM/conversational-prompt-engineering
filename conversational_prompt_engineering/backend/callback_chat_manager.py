from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from genai.schema import ChatRole

from conversational_prompt_engineering.backend.chat_manager_util import ChatManagerBase


class CallbackChatManager(ChatManagerBase):
    def __init__(self, bam_api_key, model, conv_id) -> None:
        super().__init__(bam_api_key, model, conv_id)

        self.model_chat = []

    def add_system_message(self, msg):
        self._add_msg(self.model_chat, ChatRole.SYSTEM, msg)

    def set_instructions(self, task_instruction, api_instruction, function2description):
        self.add_system_message(task_instruction)
        self.add_system_message(api_instruction)
        for fun_sign, fun_descr in function2description.items():
            self.add_system_message(f'function {fun_sign}: {fun_descr}')

    def submit_model_chat_and_process_response(self):
        resp = self._get_assistant_response(self.model_chat)
        self._add_msg(self.model_chat, ChatRole.ASSISTANT, resp)
        exec(resp.replace('\n', '\\n'))


class TestManager(CallbackChatManager):
    def __init__(self, bam_api_key, model, conv_id) -> None:
        super().__init__(bam_api_key, model, conv_id)
        self.user_chat = []
        self.examples = None
        self.prompts = []

    def add_user_message(self, message):
        self._add_msg(self.user_chat, ChatRole.USER, message)
        self._add_msg(self.model_chat, ChatRole.USER, message)
        self.submit_model_chat_and_process_response()

    def submit_message_to_user(self, message):
        self._add_msg(self.user_chat, ChatRole.ASSISTANT, message)

    def submit_prompt(self, prompt):
        self.prompts.append(prompt)

        futures = {}
        with ThreadPoolExecutor(max_workers=len(self.examples)) as executor:
            for i, example in enumerate(self.examples):
                tmp_chat = []
                self._add_msg(tmp_chat, ChatRole.SYSTEM, prompt + '\Text: ' + example + '\nSummary: ')
                futures[i] = executor.submit(self._get_assistant_response, tmp_chat)

        self.add_system_message(f'The suggested prompt has produced the following summaries for the user examples:')
        for i, f in futures.items():
            summary = f.result()
            self.add_system_message(f'Example {i + 1}: {summary}')

        if len(self.prompts) == 1:
            self.add_system_message(
                'Present the summaries for the examples one by one and discuss them with the user. '
                'Ask the user if they want to see the original text. '
                'You dont have to go through all the examples, '
                'when you have gathered enough feedback to suggest a new prompt - submit it.'
                'Remember to communicate only via API calls.'
            )
        else:
            self.add_system_message(
                'Do ALL the produced summaries satisfy the user comments and the approved summaries from the previous discussion? '
                'If not, suggest a better prompt via submit_prompt call directly, without involving the user.'
                "If ALL the produced summaries satisfy the previous discussion, discuss them with the user, one by one."
                'Remember to communicate only via API calls.'
            )
        self.submit_model_chat_and_process_response()

    def submit_summary_feedback(self, feedback):
        self.add_system_message('Suggest a new prompt that would take into account the user feedback.')
        self.submit_model_chat_and_process_response()

    def init_chat(self, df, max_num_examples=3):
        task_instruction = \
            'You and I (system) will work together to build a prompt for summarization task for the user.' \
            'You will interact with the user to gather information, and discuss the summaries. ' \
            'I will generate the summaries from the prompts you suggest, and pass them back to you, ' \
            'so that you could discuss them with the user.'
        api_instruction = \
            'You should communicate with the user and system ONLY via python API described below, and not via direct messages. ' \
            'Format ALL your answers python code calling one of the following functions:'
        api = {
            'self.submit_message_to_user(message)': 'call this function to submit your message to the user',
            'self.submit_prompt(prompt)': 'call this function to inform the system that you have a suggestion for the prompt',
            'self.submit_summary_feedback(feedback)': 'call this function to submit the user feedback to the summary generated with your last prompt'
        }
        self.set_instructions(task_instruction, api_instruction, api)

        self.examples = df['text'].sample(max_num_examples).tolist()
        self.add_system_message('The user has provided the following examples for the texts to summarize, '
                                'briefly discuss them with the user before suggesting the prompt. '
                                'Your suggestion should take into account the user comments and corrections.')
        for i, ex in enumerate(self.examples):
            self.add_system_message(f'Example {i + 1}: {ex}')

        self.submit_model_chat_and_process_response()


def delme_test():
    mgr = TestManager('pak-Q-b9JJQlQYaVH3gS_KCY_ObSMRv3HTNAHUp_XIzWbyY', "llama3", 'a')
    mgr.init_chat(pd.read_csv(
        '/Users/artemspector/PycharmProjects/ace/conversational-prompt-engineering/conversational_prompt_engineering/data/legal_plain_english/train.csv'))

    while True:
        try:
            from_idx = len(mgr.user_chat) - list(reversed([msg['role'] for msg in mgr.user_chat])).index(ChatRole.USER)
        except ValueError:
            from_idx = 0
        for msg in mgr.user_chat[from_idx:]:
            print(f'{msg["role"]}: {msg["content"]}')
        user_msg = input('user: ')
        mgr.add_user_message(user_msg)


if __name__ == '__main__':
    delme_test()
