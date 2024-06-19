from concurrent.futures import ThreadPoolExecutor

from genai.schema import ChatRole

from conversational_prompt_engineering.backend.chat_manager_util import ChatManagerBase


class ModelPrompts:
    def __init__(self) -> None:
        self.task_instruction = \
            'You and I (system) will work together to build a prompt for the task of the user via a chat with the user.' \
            'This prompt will be fed to a model dedicated to perform the user\'s task.' \
            'Our aim is to build a prompt that when fed to the model, produce outputs that are aligned with the user\'s expectations.' \
            'Thus, the prompt should reflect the specific requirements and preferences of the user ' \
            'from the output as expressed in the chat.'\
            'You will interact with the user to gather information regarding their preferences and needs. ' \
            'I will send the prompts you suggest to the dedicated model to generate outputs, and pass them back to you, ' \
            'so that you could discuss them with the user and get feedback. ' \
            'User time is valuable, keep the conversation pragmatic. Make the obvious decisions by yourself.' \
            'Don\'t greet the user at your first interaction.'

        self.api_instruction = \
            'You should communicate with the user and system ONLY via python API described below, and not via direct messages. ' \
            'The input parameters to API functions are strings. Enclose them in double quotes, and escape all double quotes inside these strings. ' \
            'Format ALL your answers python code calling one of the following functions:'

        self.api = {
            'self.submit_message_to_user(message)': 'call this function to submit your message to the user. Use markdown to mark the prompts and the outputs.',
            'self.submit_prompt(prompt)': 'call this function to inform the system that you have a new suggestion for the prompt',
            'self.output_accepted(example_num, output)': 'call this function every time the user accepts an output. Pass the example number and the output text as parameters.',
            'self.end_outputs_discussion()': 'call this function after all the outputs have been discussed with the user.',
            'self.conversation_end()': 'call this function when the user wants to end the conversation.',
        }

        self.examples_intro = 'Here are some examples of the input texts provided by the user:'

        self.examples_instruction = \
            'Start with asking the user which task they would like to perform. ' \
            'Then, before suggesting the prompt, briefly discuss the text examples with the user and ask them relevant questions regarding their output requirements and preferences.  ' \
            'Your suggested prompt should reflect the user\'s expectations from the task output as expressed during the chat.' \
            'Share the suggested prompt with the user before submitting it.' \
            'Remember to communicate only via API calls.'


        self.result_intro = 'Based on the suggested prompt, the model has produced the following outputs for the user input examples:'

        self.analyze_result_instruction = \
            'For each example show the full model output to the user and discuss it with them, one example at a time. ' \
            'Note that the user has not seen these outputs yet, when presenting an output show its full text.\n' \
            'The discussion should result in an output accepted by the user.\n' \
            'When the user accepts an output (directly or indirectly), call output_accepted API passing the example number and the output text. ' \
            'Continue your conversation with the user in any case.\n' \
            'After all the outputs were accepted by the user, call end_outputs_discussion.\n' \
            'Remember to communicate only via API calls.'

        self.syntax_err_instruction = 'The last API call produced a syntax error. Return the same call with fixed error.'


class CallbackChatManager(ChatManagerBase):
    def __init__(self, bam_api_key, model, conv_id) -> None:
        super().__init__(bam_api_key, model, conv_id)
        self.model_prompts = ModelPrompts()

        self.api_names = None

        self.model_chat = []
        self.model_chat_length = 0
        self.user_chat = []
        self.user_chat_length = 0

        self.dataset_name = None
        self.enable_upload_file = True

        self.examples = None
        self.outputs = None
        self.prompts = []

        self.output_discussion_state = None

    @property
    def approved_prompts(self):
        return [{'prompt': p} for p in self.prompts]

    @property
    def approved_outputs(self):
        return [{'text': t, 'output': s} for t, s in zip(self.examples, self.outputs) if s is not None]

    @property
    def validated_example_idx(self):
        return len([s for s in self.outputs if s is not None])

    def add_system_message(self, msg):
        self._add_msg(self.model_chat, ChatRole.SYSTEM, msg)

    def submit_model_chat_and_process_response(self):
        if len(self.model_chat) > self.model_chat_length:
            resp = self._get_assistant_response(self.model_chat)
            self._add_msg(self.model_chat, ChatRole.ASSISTANT, resp)
            self.model_chat_length = len(self.model_chat)
            api_indices = sorted([resp.index(name) for name in self.api_names if name in resp])
            api_calls = [resp[begin: end].strip() for begin, end in zip(api_indices, api_indices[1:] + [len(resp)])]
            for call in api_calls:
                escaped_call = call.replace('\n', '\\n')
                try:
                    exec(escaped_call)
                except SyntaxError:
                    self.add_system_message(self.model_prompts.syntax_err_instruction)
                    self.submit_model_chat_and_process_response()

    def add_user_message(self, message):
        self._add_msg(self.user_chat, ChatRole.USER, message)
        self.user_chat_length = len(self.user_chat)  # user message is rendered by cpe
        self._add_msg(self.model_chat, ChatRole.USER, message)

    def add_welcome_message(self):
        static_assistant_hello_msg = ["Hello! I'm an IBM prompt building assistant. I'm here to help you build an effective instruction for your task.\n",
                                      "We'll work together to craft a prompt that yields high-quality results that are aligned with your output preferences. \n"
                                      "Here's an overview of our collaboration:\n",
                                      "\n1. I'll first ask you to share a short description of your task."
                                      "\n2. I'll then ask you to share some typical input texts for the task, without their outputs, and will use them to generate an initial dedicated prompt."
                                      "\n3. I'll refine the prompt based on your feedback on my generated outputs."
                                      "\n4. Finally, I'll share the resultant few-shot prompt."
                                      "Once we've built a prompt, you can evaluate its performance by clicking on \"Evaluate\" on the side-bar.\n",
                                      "To get started, could you please upload your data."]

        self._add_msg(chat=self.user_chat, role=ChatRole.ASSISTANT, msg="\n".join(static_assistant_hello_msg))

    def generate_agent_messages(self):
        self.submit_model_chat_and_process_response()
        agent_messages = []
        if len(self.user_chat) > self.user_chat_length:
            for msg in self.user_chat[self.user_chat_length:]:
                if msg['role'] == ChatRole.ASSISTANT:
                    agent_messages.append(msg)
            self.user_chat_length = len(self.user_chat)
        self.save_chat_html(self.user_chat, "user_chat.html")
        self.save_chat_html(self.model_chat, "model_chat.html")
        return agent_messages

    def submit_message_to_user(self, message):
        self._add_msg(self.user_chat, ChatRole.ASSISTANT, message)

    def submit_prompt(self, prompt):
        self.prompts.append(prompt)

        futures = {}
        with ThreadPoolExecutor(max_workers=len(self.examples)) as executor:
            for i, example in enumerate(self.examples):
                tmp_chat = []
                self._add_msg(tmp_chat, ChatRole.SYSTEM, prompt + '\Text: ' + example + '\nOutput: ')
                futures[i] = executor.submit(self._get_assistant_response, tmp_chat)

        self.output_discussion_state = {
            'model_outputs': [None] * len(self.examples),
            'user_chat_begin': self.user_chat_length
        }
        self.add_system_message(self.model_prompts.result_intro)
        for i, f in futures.items():
            output = f.result()
            self.add_system_message(f'Example {i + 1}: {output}')
            self.output_discussion_state['model_outputs'][i] = output

        self.add_system_message(self.model_prompts.analyze_result_instruction)

        self.submit_model_chat_and_process_response()

    def output_accepted(self, example_num, output):
        example_idx = int(example_num) - 1
        self.outputs[example_idx] = output

    def end_outputs_discussion(self):
        analyze_discussion_task = \
            f'You need to decide whether the outputs generated by the model from the prompt "{self.prompts[-1]}" ' \
            f'were accepted by the user as-is or needed some adjustments. ' \
            f'Compare the model outputs to the accepted ones and give recommendations for the prompt improvement so that it would produce the accepted outputs directly.'

        analyze_discussion_user_comments = 'Here are the user comments about the model outputs:\n'

        analyze_discussion_continue = 'Continue your conversation with the user taking into account these recommendations above.'

        temp_chat = []
        self._add_msg(temp_chat, ChatRole.SYSTEM, analyze_discussion_task)
        txt_produced_accepted = zip(self.examples, self.output_discussion_state['model_outputs'], self.outputs)
        for i, (txt, produced, accepted) in enumerate(txt_produced_accepted):
            example_txt = f'Example {i + 1}\nText:{txt}\nModel output:{produced}\nAccepted output:{accepted}'
            self._add_msg(temp_chat, ChatRole.SYSTEM, example_txt)

        self._add_msg(temp_chat, ChatRole.SYSTEM, analyze_discussion_user_comments)
        temp_chat += [msg for msg in self.user_chat[self.output_discussion_state['user_chat_begin']:]
                      if msg['role'] == ChatRole.USER]

        recommendations = self._get_assistant_response(temp_chat)
        self.add_system_message(recommendations + '\n' + analyze_discussion_continue)
        self.output_discussion_state = None
        self.submit_model_chat_and_process_response()

    def conversation_end(self):
        # placeholder
        pass

    def set_instructions(self, task_instruction, api_instruction, function2description):
        self.api_names = [key[:key.index('(')] for key in function2description.keys()]
        self.add_system_message(task_instruction)
        self.add_system_message(api_instruction)
        for fun_sign, fun_descr in function2description.items():
            self.add_system_message(f'function {fun_sign}: {fun_descr}')

    def init_chat(self, examples):
        self.set_instructions(self.model_prompts.task_instruction, self.model_prompts.api_instruction,
                              self.model_prompts.api)

        self.outputs = [None] * len(examples)
        self.examples = examples

        self.add_system_message(self.model_prompts.examples_intro)
        for i, ex in enumerate(self.examples):
            self.add_system_message(f'Example {i + 1}: {ex}')

        self.add_system_message(self.model_prompts.examples_instruction)

        self.submit_model_chat_and_process_response()

    def process_examples(self, df, dataset_name):
        self.dataset_name = dataset_name
        self.enable_upload_file = False
        examples = df['text'].tolist()[:3]
        self.init_chat(examples)
