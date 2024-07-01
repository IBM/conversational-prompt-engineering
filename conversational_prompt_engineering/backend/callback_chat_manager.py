import logging
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
            'from the output as expressed in the chat.' \
            'You will interact with the user to gather information regarding their preferences and needs. ' \
            'I will send the prompts you suggest to the dedicated model to generate outputs, and pass them back to you, ' \
            'so that you could discuss them with the user and get feedback. ' \
            'User time is valuable, keep the conversation pragmatic. Make the obvious decisions by yourself.' \
            'Don\'t greet the user at your first interaction.'

        self.api_instruction = \
            'You should communicate with the user and system ONLY via python API described below, and not via direct messages. ' \
            'The input parameters to API functions are strings. Enclose them in double quotes, and escape all double quotes inside these strings to avoid syntax errors. ' \
            'Note that the user is not aware of the API, so don\'t not tell the user which API you are going to call.\n' \
            'Format ALL your answers python code calling one of the following functions:'

        self.api = {
            'self.submit_message_to_user(message)': 'call this function to submit your message to the user. Use markdown to mark the prompts and the outputs.',
            'self.submit_prompt(prompt)': 'call this function to inform the system that you have a new suggestion for the prompt. Use it only with the prompts approved by the user.',
            'self.switch_to_example(example_num)': 'call this function before you start discussing with the user an output of a specific example, and pass the example number as parameter.',
            'self.show_original_text(example_num)': 'call this function when the user asks to show the original text of an example, and pass the example number as parameter',
            'self.output_accepted(example_num, output)': 'call this function every time the user accepts an output. Pass the example number and the output text as parameters.',
            'self.end_outputs_discussion()': 'call this function after all the outputs have been discussed with the user and all NUM_EXAMPLES outputs were accepted by the user.',
            'self.conversation_end()': 'call this function when the user wants to end the conversation.',
            'self.task_is_defined()': 'call this function when the user has defined the task and it\'s clear to you. You should only use this callback once'
        }

        self.discuss_example_num = 'Discuss with the user the output of Example '

        self.examples_intro = 'Here are some examples of the input texts provided by the user:'

        self.task_definition_instruction = \
            'Start with asking the user which task they would like to perform on the texts. ' \
            'Once the task is clear to you, call task_is_defined API.'

        self.analyze_examples = \
            'Before suggesting the prompt, briefly discuss the text examples with the user and ask them relevant questions regarding their output requirements and preferences. Please take into account the specific characteristics of the data. ' \
            'Your suggested prompt should reflect the user\'s expectations from the task output as expressed during the chat.' \
            'Share the suggested prompt with the user before submitting it.' \
            'Remember to communicate only via API calls.'\
            'From this point, don\'t use task_is_defined API'

        self.generate_baseline_instruction_task = \
            'After the user has provided the task description and the examples, generate a general prompt for this task'

        self.result_intro = 'Based on the suggested prompt, the model has produced the following outputs for the user input examples:'

        self.analyze_result_instruction = \
            'For each of NUM_EXAMPLES examples show the full model output to the user and discuss it with them, one example at a time. ' \
            'Note that the user has not seen these outputs yet, when presenting an output show its full text.\n' \
            'The discussion should result in an output accepted by the user.\n' \
            'When the user asks to show the original text of an example, call show_original_text API passing the example number.\n' \
            'When the user accepts an output (directly or indirectly), call output_accepted API passing the example number and the output text. ' \
            'when the user asks to update the prompt, share the prompt with him.\n' \
            'Continue your conversation with the user after they accept the output.\n' \
            'Remember to communicate only via API calls.'

        self.syntax_err_instruction = 'The last API call produced a syntax error. Return the same call with fixed error.'
        self.api_only_instruction = 'Please communicate only via API calls defined above. Do not use plain text or non-existing API in the response.'

        self.analyze_discussion_task = \
            'You need to decide whether the outputs generated by the model from the prompt "PROMPT" ' \
            'were accepted by the user as-is or needed some adjustments. ' \
            'In case the user asked for adjustments, compare the model outputs to the accepted ones ' \
            'and give recommendations for the prompt improvement so that it would produce the accepted outputs directly.'

        self.analyze_discussion_user_comments = 'Here are the user comments about the model outputs:\n'

        self.analyze_discussion_continue = 'Continue your conversation with the user taking into account these recommendations above. ' \
                                           'If the prompt should be modified based on these recommendations, then present it to the user.'


class MixtralPrompts(ModelPrompts):
    def __init__(self) -> None:
        super().__init__()


class Llama3Prompts(ModelPrompts):
    def __init__(self) -> None:
        super().__init__()


class CallbackChatManager(ChatManagerBase):
    def __init__(self, bam_api_key, model, conv_id) -> None:
        super().__init__(bam_api_key, model, conv_id)
        self.model_prompts = {
            'mixtral': MixtralPrompts,
            'llama-3': Llama3Prompts,
        }[model]()

        self.api_names = None

        self.model_chat = []
        self.model_chat_length = 0
        self.example_num = None
        self.user_chat = []
        self.user_chat_length = 0

        self.dataset_name = None
        self.enable_upload_file = True

        self.examples = None
        self.outputs = None
        self.prompts = []
        self.baseline_prompt = ""
        self.user_default_prompt = ""
        self.user_session_name = ""

        self.output_discussion_state = None
        self.calls_queue = []
        self.call_depth = 0


    @property
    def approved_prompts(self):
        return [{'prompt': p} for p in self.prompts]

    @property
    def approved_outputs(self):
        return [{'text': t, 'summary': s} for t, s in zip(self.examples, self.outputs) if s is not None]

    @property
    def validated_example_idx(self):
        return len([s for s in self.outputs if s is not None])

    def _add_msg(self, chat, role, msg, example_num=None):
        message = {'role': role, 'content': msg}
        if example_num is not None:
            message['example_num'] = example_num
        chat.append(message)

    def add_system_message(self, msg, example_num=None):
        self._add_msg(self.model_chat, ChatRole.SYSTEM, msg, example_num)

    @property
    def _filtered_model_chat(self):
        return [msg for msg in self.model_chat
                if self.example_num is None or msg.get('example_num', self.example_num) == self.example_num]

    def submit_model_chat_and_process_response(self):
        max_call_depth = 3
        if self.call_depth >= max_call_depth:
            return
        else:
            self.call_depth += 1

        execute_calls = len(self.calls_queue) == 0
        if len(self.model_chat) > self.model_chat_length:
            resp = self._get_assistant_response(self._filtered_model_chat)
            self._add_msg(self.model_chat, ChatRole.ASSISTANT, resp)
            if resp.startswith('```python\n'):
                resp = resp[len('```python\n'): -len('\n```')]
            self.model_chat_length = len(self.model_chat)
            api_indices = sorted([resp.index(name) for name in self.api_names if name in resp])
            api_calls = [resp[begin: end].strip().replace('\n', '\\n')
                         for begin, end in zip(api_indices, api_indices[1:] + [len(resp)])]
            if len(api_indices) == 0 or api_indices[0] > 2:
                self.add_system_message(self.model_prompts.api_only_instruction)
                self.submit_model_chat_and_process_response()
            else:
                self.calls_queue += api_calls

        if execute_calls:
            while len(self.calls_queue) > 0:
                call = self.calls_queue.pop(0)
                try:
                    exec(call)
                except SyntaxError:
                    self.calls_queue = []
                    self.add_system_message(self.model_prompts.syntax_err_instruction)
                    self.submit_model_chat_and_process_response()

        self.call_depth -= 1

    def add_user_message(self, message):
        self._add_msg(self.user_chat, ChatRole.USER, message)
        self.user_chat_length = len(self.user_chat)  # user message is rendered by cpe
        self._add_msg(self.model_chat, ChatRole.USER, message)  # not adding dummy initial user message

    def add_user_message_only_to_user_chat(self, message):
        self._add_msg(self.user_chat, ChatRole.USER, message)
        self.user_chat_length = len(self.user_chat)  # user message is rendered by cpe

    def add_welcome_message(self):
        static_assistant_hello_msg = [
            "Hello! I'm an IBM prompt building assistant. In the following session we will work together through a natural conversation, to build an effective instruction – a.k.a. prompt – personalized for your task and data.",
            "\nTo begin, please upload your data, or select a dataset from our datasets catalog above."]

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
        if self.example_num is not None:
            self.save_chat_html(self._filtered_model_chat, f'model_chat_example_{self.example_num}.html')
        if self.outputs:
            self.save_prompts_and_config(self.approved_prompts, self.approved_outputs)
        return agent_messages

    def submit_message_to_user(self, message):
        self._add_msg(self.user_chat, ChatRole.ASSISTANT, message)

    def show_original_text(self, example_num):
        txt = self.examples[int(example_num) - 1]
        self._add_msg(chat=self.user_chat, role=ChatRole.ASSISTANT, msg=txt)
        self.add_system_message(f'The original text for Example {example_num} was shown to the user.')

    def task_is_defined(self):
        # open side chat with model
        assert self.baseline_prompt == "", "second callback to task_is_defined!"
        self.calls_queue = []

        tmp_chat = self.model_chat[:]
        self._add_msg(tmp_chat, ChatRole.SYSTEM, self.model_prompts.generate_baseline_instruction_task)
        resp = self._get_assistant_response(tmp_chat)
        self.baseline_prompt = resp[:-2].replace("self.submit_prompt(\"", "")
        logging.info(f"baseline prompt is {self.baseline_prompt}")
        self.add_system_message(self.model_prompts.analyze_examples)
        self.submit_model_chat_and_process_response()

    def _strip_user_message(self):
        last_msg = self.model_chat[-1]
        submit_message_to_user = 'self.submit_message_to_user'
        if submit_message_to_user in last_msg['content']:
            last_msg['content'] = last_msg['content'][:last_msg['content'].index(submit_message_to_user)]

    def switch_to_example(self, example_num):
        example_num = int(example_num)
        self.example_num = example_num
        self.calls_queue = []
        self._strip_user_message()
        discuss_ex = self.model_prompts.discuss_example_num + str(self.example_num)
        self.add_system_message(discuss_ex)
        self.submit_model_chat_and_process_response()

    def submit_prompt(self, prompt):
        self.calls_queue = []
        self.prompts.append(prompt)

        futures = {}
        with ThreadPoolExecutor(max_workers=len(self.examples)) as executor:
            for i, example in enumerate(self.examples):
                tmp_chat = []
                self._add_msg(tmp_chat, ChatRole.SYSTEM, prompt + '\nText: ' + example + '\nOutput: ')
                futures[i] = executor.submit(self._get_assistant_response, tmp_chat)

        self.output_discussion_state = {
            'model_outputs': [None] * len(self.examples),
            'user_chat_begin': self.user_chat_length
        }
        self.add_system_message(self.model_prompts.result_intro)
        for i, f in futures.items():
            output = f.result()
            example_num = i + 1
            self.add_system_message(f'Example {example_num}: {output}', example_num)
            self.output_discussion_state['model_outputs'][i] = output

        self.add_system_message(
            self.model_prompts.analyze_result_instruction.replace('NUM_EXAMPLES', str(len(self.examples))))
        self.submit_model_chat_and_process_response()

    def output_accepted(self, example_num, output):
        example_idx = int(example_num) - 1
        self.outputs[example_idx] = output

    def end_outputs_discussion(self):
        self.calls_queue = []
        temp_chat = []
        self._add_msg(temp_chat, ChatRole.SYSTEM,
                      self.model_prompts.analyze_discussion_task.replace('PROMPT', self.prompts[-1]))
        txt_produced_accepted = zip(self.examples, self.output_discussion_state['model_outputs'], self.outputs)
        for i, (txt, produced, accepted) in enumerate(txt_produced_accepted):
            example_txt = f'Example {i + 1}\nText:{txt}\nModel output:{produced}\nAccepted output:{accepted}'
            self._add_msg(temp_chat, ChatRole.SYSTEM, example_txt)

        self._add_msg(temp_chat, ChatRole.SYSTEM, self.model_prompts.analyze_discussion_user_comments)
        temp_chat += [msg for msg in self.user_chat[self.output_discussion_state['user_chat_begin']:]
                      if msg['role'] == ChatRole.USER]

        recommendations = self._get_assistant_response(temp_chat)
        self.add_system_message(recommendations + '\n' + self.model_prompts.analyze_discussion_continue)
        self.output_discussion_state = None
        self.submit_model_chat_and_process_response()

    def conversation_end(self):
        # placeholder
        pass

    def set_instructions(self, task_instruction, api_instruction, function2description):
        self.api_names = [key[:key.index('(')] for key in function2description.keys()]
        self.add_system_message(task_instruction)
        self.add_system_message(api_instruction)
        num_examples = str(len(self.examples))
        for fun_sign, fun_descr in function2description.items():
            self.add_system_message(f'function {fun_sign}: {fun_descr.replace("task_is_defined", num_examples)}')

    def init_chat(self, examples):
        self.outputs = [None] * len(examples)
        self.examples = examples

        self.set_instructions(self.model_prompts.task_instruction, self.model_prompts.api_instruction,
                              self.model_prompts.api)

        self.add_system_message(self.model_prompts.examples_intro)
        for i, ex in enumerate(self.examples):
            example_num = i + 1
            self.example_num = example_num
            self.add_system_message(f'Example {example_num}: {ex}', example_num)
        self.example_num = None

        self.add_system_message(self.model_prompts.task_definition_instruction)

        self.submit_model_chat_and_process_response()

    def process_examples(self, df, dataset_name):
        self.dataset_name = dataset_name
        self.enable_upload_file = False
        examples = df['text'].tolist()[:3]
        self.init_chat(examples)
