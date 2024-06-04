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
        exec(resp)


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
        self._add_msg(self.model_chat, ChatRole.ASSISTANT, message)

    def submit_prompt(self, prompt):
        # TODO: create summaries for all the examples simultaneously
        tmp_chat = []
        text = self.examples[0]
        self._add_msg(tmp_chat, ChatRole.SYSTEM, prompt + '\Text: ' + text + '\nSummary: ')
        summary = self._get_assistant_response(tmp_chat)

        prompt_res = {'prompt': prompt, 'text': text, 'summary': summary}
        if len(self.prompts) == 0 or 'feedback' in self.prompts[-1]:
            self.prompts.append(prompt_res)
        else:
            self.prompts[-1] = prompt_res

        self.add_system_message(f'The prompt "{prompt}" has produced for the text "{text}" the summary "{summary}".')
        if len(self.prompts) > 1:
            self.add_system_message('Make sure the result is consistent with the previous user feedbacks. '
                                    'If it is - present the result to the user and collect their feedback, '
                                    'otherwise  suggest a better prompt.')
        else:
            self.add_system_message('Discuss the summary with the user, and when done submit the collected feedback. '
                                    'Remember to communicate only via API calls.')
        self.submit_model_chat_and_process_response()

    def submit_summary_feedback(self, feedback):
        self.prompts[-1]['feedback'] = feedback
        raise NotImplementedError

    def init_chat(self, df):
        task_instruction = \
            'Your task is build a prompt for summarization task for the user. ' \
            'You will interact with the user to gather information, demonstrate the summaries, and get the feedback. ' \
            'You will also interact with the System (me) to report the progress, and keep records of the process.'
        api_instruction = \
            'You should communicate with the user and system only via python API described below, and not via direct messages. ' \
            'Format all your answers as python code calling one of the following functions:'
        api = {
            'self.submit_message_to_user(message)': 'call this function to submit your message to the user',
            'self.submit_prompt(prompt)': 'call this function to inform the system that you have a suggestion for the prompt',
            'self.submit_summary_feedback(feedback)': 'call this function to submit the user feedback to the summary generated with your last prompt'
        }
        self.set_instructions(task_instruction, api_instruction, api)

        self.examples = df['text'].sample(5).tolist()
        self.add_system_message('The user has provided the following examples for the texts to summarize, '
                                'briefly discuss them with the user before suggesting the prompt.')
        for i, ex in enumerate(self.examples):
            self.add_system_message(f'{i + 1}. {ex}')

        self.submit_model_chat_and_process_response()


def delme_test():
    mgr = TestManager('pak-Q-b9JJQlQYaVH3gS_KCY_ObSMRv3HTNAHUp_XIzWbyY', "llama3", 'a')
    mgr.init_chat(pd.read_csv(
        '/Users/artemspector/PycharmProjects/ace/conversational-prompt-engineering/conversational_prompt_engineering/data/legal_plain_english/train.csv'))

    while True:
        print(f'{mgr.user_chat[-1]["role"]}: {mgr.user_chat[-1]["content"]}')
        user_msg = input('user: ')
        mgr.add_user_message(user_msg)


if __name__ == '__main__':
    delme_test()
