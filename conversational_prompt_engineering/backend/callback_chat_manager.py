from concurrent.futures import ThreadPoolExecutor

from genai.schema import ChatRole

from conversational_prompt_engineering.backend.chat_manager_util import ChatManagerBase


class CallbackChatManager(ChatManagerBase):
    def __init__(self, bam_api_key, model, conv_id) -> None:
        super().__init__(bam_api_key, model, conv_id)

        self.model_chat = []
        self.model_chat_length = 0
        self.user_chat = []
        self.user_chat_length = 0

        self.dataset_name = None
        self.enable_upload_file = True

        self.examples = None
        self.prompts = []
        self.next_instruction = None

    def add_system_message(self, msg):
        self._add_msg(self.model_chat, ChatRole.SYSTEM, msg)

    def set_instructions(self, task_instruction, api_instruction, function2description):
        self.add_system_message(task_instruction)
        self.add_system_message(api_instruction)
        for fun_sign, fun_descr in function2description.items():
            self.add_system_message(f'function {fun_sign}: {fun_descr}')

    def submit_model_chat_and_process_response(self):
        if len(self.model_chat) > self.model_chat_length:
            resp = self._get_assistant_response(self.model_chat)
            self._add_msg(self.model_chat, ChatRole.ASSISTANT, resp)
            self.model_chat_length = len(self.model_chat)
            escaped_resp = resp.replace('\n', '\\n').replace('\\n\\nself.', '\n\nself.')
            exec(escaped_resp)

    def add_user_message(self, message):
        self._add_msg(self.user_chat, ChatRole.USER, message)
        self.user_chat_length = len(self.user_chat)  # user message is rendered by cpe
        self._add_msg(self.model_chat, ChatRole.USER, message)

    def generate_agent_messages(self):
        self.submit_model_chat_and_process_response()
        agent_messages = []
        if len(self.user_chat) > self.user_chat_length:
            for msg in self.user_chat[self.user_chat_length:]:
                if msg['role'] == ChatRole.ASSISTANT:
                    agent_messages.append(msg)
            self.user_chat_length = len(self.user_chat)

        return agent_messages

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
                'Summarize the user comments and approved summaries from the discussion for each example. '
                'Reply to the system and not to the user.'
                'Remember to communicate only via API calls.'
            )
            self.next_instruction = \
                'Compare the produced summaries to the approved ones. Decide whether the prompt is good or should be improved. ' \
                'If the prompt should be improved - suggest a better prompt. Notify the user via submit_message_to_user, and submit it via submit_prompt.\n' \
                'If the prompt is good - discuss the produced summaries with the user via submit_message_to_user, one example at a time.\n' \
                'Remember to communicate only via API calls.'
        self.submit_model_chat_and_process_response()

    def submit_message_to_system(self, message):
        if self.next_instruction is not None:
            self.add_system_message(self.next_instruction)
            self.next_instruction = None
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
            'self.submit_message_to_system(message)': 'call this function to submit your message to system (me) when I ask you a direct question',
            'self.submit_prompt(prompt)': 'call this function to inform the system that you have a suggestion for the prompt',
        }
        self.set_instructions(task_instruction, api_instruction, api)

        self.examples = df['text'].sample(max_num_examples).tolist()
        self.add_system_message('The user has provided the following examples for the texts to summarize, '
                                'briefly discuss them with the user before suggesting the prompt. '
                                'Your suggestion should take into account the user comments and corrections.'
                                'Remember to communicate only via API calls.')
        for i, ex in enumerate(self.examples):
            self.add_system_message(f'Example {i + 1}: {ex}')

        self.submit_model_chat_and_process_response()

    def process_examples(self, df, dataset_name):
        self.dataset_name = dataset_name
        self.enable_upload_file = False
        self.init_chat(df)

