import json
import logging
import os.path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from genai.schema import ChatRole

from conversational_prompt_engineering.backend.chat_manager_util import ChatManagerBase
from conversational_prompt_engineering.backend.prompt_building_util import build_few_shot_prompt
from conversational_prompt_engineering.data.main_dataset_name_to_dir import dataset_name_to_dir


class ModelPrompts:
    def __init__(self) -> None:
        self.task_instruction = \
            'You and I (system) will work together to build a prompt for the task of the user via a chat with the user. ' \
            'This prompt will be fed to a model dedicated to perform the user\'s task. ' \
            'Our aim is to build a prompt that when fed to the model, produce outputs that are aligned with the user\'s expectations. ' \
            'Thus, the prompt should reflect the specific requirements and preferences of the user ' \
            'from the output as expressed in the chat. ' \
            'You will interact with the user to gather information regarding their preferences and needs. ' \
            'I will send the prompts you suggest to the dedicated model to generate outputs, and pass them back to you, ' \
            'so that you could discuss them with the user and get feedback. ' \
            'User time is valuable, keep the conversation pragmatic. Make the obvious decisions by yourself. ' \
            'Don\'t greet the user at your first interaction.'

        self.api_instruction = \
            'You should communicate with the user and system ONLY via python API described below, and not via direct messages. ' \
            'The input parameters to API functions should be string literals using double quotes. Remember to escape double-quote characters inside the parameter values.\n' \
            'Note that the user is not aware of the API, so don\'t not tell the user which API you are going to call.\n ' \
            'Format ALL your answers python code calling one of the following functions:'

        self.api = {
            'self.submit_message_to_user(message)': 'call this function to submit your message to the user. Use markdown to mark the prompts and the outputs. ',
            'self.submit_prompt(prompt)': 'call this function to inform the system that you have a new suggestion for the prompt. Use it only with the prompts approved by the user. ',
            'self.switch_to_example(example_num)': 'call this function before you start discussing with the user an output of a specific example, and pass the example number as parameter. ',
            'self.show_original_text(example_num)': 'call this function when the user asks to show the original text of an example, and pass the example number as parameter. ',
            'self.output_accepted(example_num, output)': 'call this function every time the user unequivocally accepts an output. Pass the example number and the output text as parameters. ',
            'self.end_outputs_discussion()': 'call this function after all the outputs have been discussed with the user and all NUM_EXAMPLES outputs were accepted by the user. ',
            'self.conversation_end()': 'call this function when the user wants to end the conversation. ',
            'self.task_is_defined()': 'call this function when the user has defined the task and it\'s clear to you. You should only use this callback once. '
        }

        self.examples_intro = \
            'The user has provided some text examples. ' \
            'I\'ve selected a few of them that you will use in the conversation. ' \
            'Note that your goal to build a generic prompt, and not for these specific examples.'

        self.task_definition_instruction = \
            'Start with asking the user which task they would like to perform on the texts. ' \
            'Once the task is clear to you, call task_is_defined API. '

        self.analyze_examples = \
            'Before suggesting the prompt, briefly discuss the text examples with the user and ask them relevant questions regarding their output requirements and preferences. Please take into account the specific characteristics of the data. ' \
            'Your suggested prompt should reflect the user\'s expectations from the task output as expressed during the chat. ' \
            'Share the suggested prompt with the user before submitting it. ' \
            'Remember to communicate only via API calls. ' \
            'From this point, don\'t use task_is_defined API. '

        self.generate_baseline_instruction_task = \
            'Generate a concise and general prompt for this task, for example "summarize this text" for summarization, or "generate questions in the following text" for question generation. Submit the general prompt via submit_prompt API. '

        self.result_intro = 'Based on the suggested prompt, the model has produced the following outputs for the user input examples:'

        self.analyze_result_instruction = \
            'For each of NUM_EXAMPLES examples show the model output to the user and discuss it with them, one example at a time. ' \
            'Use switch_example API to navigate between examples. ' \
            'When the user asks to show the original text of an example, call show_original_text API passing the example number.\n ' \
            'The discussion should take as long as necessary and result in an output accepted by the user in clear way, ' \
            'with no doubts, conditions or modifications. ' \
            'When the output is accepted, call output_accepted API passing the example number and the output text. ' \
            'After calling output_accepted call either switch_to_example API to move to the next example, ' \
            'or end_outputs_discussion API if all NUM_EXAMPLES have been accepted.\n' \
            'Assume that the user comments relay to the output. ' \
            'Only when the user explicitly says that he wants to update the prompt and not the output, show the updated prompt to them.\n' \
            'Remember to communicate only via API calls.'

        self.discuss_example_num = \
            'You have switched to EXAMPLE_NUM. ' \
            'Look at the user comments and the accepted outputs for the previous examples, ' \
            'apply them to the model output of this example, and present the result to the user. ' \
            'Indicate the example (number), and format the text so that the output and your text are separated by empty lines. ' \
            'Discuss the presented output taking into account the system conclusion for this example if exists.'

        self.syntax_err_instruction = \
            'The last API call produced syntax error: ERROR. Try again and fix it. ' \
            'Points to note: the call parameter must be a single string; double quotes within the string should be escaped.'

        self.api_only_instruction = \
            'Your last response is invalid because it contains some plain text or non-existing API. ' \
            'All the communications should be done as plain API calls. Try again.'

        self.analyze_discussion_task_begin = \
            'In the following discussion, the user was asked to give feedback on the model\'s outputs that were generated by the prompt "PROMPT".' \
            'The outputs that did not meet the user\'s requirements were modified.'

        self.analyze_discussion_task_end = \
            'Analyze the conversation above, and determine whether any of the outputs generated by the prompt were modified based on the user\'s feedback.' \
            'If no modifications were made, the prompt should be accepted. ' \
            'If modifications were required, recommend how to improve the prompt so that it would produce the accepted outputs directly.'

        self.analyze_discussion_continue = \
            'Continue your conversation with the user. Do the recommendations above suggest improvements to the prompt? ' \
            'If so, present the modified prompt to the user, and submit it only after the user approve it. ' \
            'Otherwise, if no modifications to the prompt are required, communicate it to user and suggest to finish the conversation.'

        self.analyze_new_prompt_task = \
            'We are working on a prompt that would produce the outputs as preferred by the user. ' \
            'Below are the outputs for some text examples which are accepted by the user, ' \
            'and also the outputs for the same examples produced by a prompt. ' \
            'Compare the accepted outputs to the produced ones for each example, ' \
            'and give your conclusion about whether the prompt produces outputs as expected by the user.\n'
        self.analyze_new_prompt_accepted_outputs = 'Accepted outputs:\n'
        self.analyze_new_prompt_new_outputs = '\nProduced outputs:\n'

        self.conversation_end_instruction = \
            'This is the end of conversation. Say goodbye to the user, ' \
            'and inform that the final prompt that includes few-shot examples and is formatted for the *TARGET_MODEL* ' \
            'can be downloaded via **Download few shot prompt** button below. ' \
            'Also, kindly refer the user to the survey tab that is now available, and let the user know that we will appreciate any feedback.'


class MixtralPrompts(ModelPrompts):
    def __init__(self) -> None:
        super().__init__()


class Llama3Prompts(ModelPrompts):
    def __init__(self) -> None:
        super().__init__()


class CallbackChatManager(ChatManagerBase):
    def __init__(self, credentials, model, conv_id, target_model, api, email_address, output_dir, config_name) -> None:
        super().__init__(credentials, model, conv_id, target_model, api, email_address, output_dir, config_name)
        self.model_prompts = {
            'mixtral': MixtralPrompts,
            'llama-3': Llama3Prompts,
        }[model]()
        self.model = model
        self.target_model = target_model

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
        self.baseline_prompts = {}
        self.prompt_conv_end = False
        self.few_shot_prompt = None

        self.output_discussion_state = None
        self.calls_queue = []
        self.cot_count = 1

    @property
    def approved_prompts(self):
        return [{'prompt': p} for p in self.prompts]

    @property
    def approved_outputs(self):
        return [{'text': t, 'summary': s} for t, s in zip(self.examples, self.outputs) if s is not None]

    @property
    def validated_example_idx(self):
        return len([s for s in self.outputs if s is not None])

    @property
    def prompt_iteration(self):
        return len(self.prompts) or None

    @property
    def _filtered_model_chat(self):
        def _include_msg(m):
            tag_values = {
                'example_num': self.example_num,
                'prompt_iteration': self.prompt_iteration
            }
            return all([curr_val is None or (m.get(name, None) or curr_val) == curr_val
                        for name, curr_val in tag_values.items()])

        return [msg for msg in self.model_chat if _include_msg(msg)]

    def _add_msg(self, chat, role, msg, **tag_kwargs):
        chat.append({'role': role, 'content': msg, **tag_kwargs})

    def add_system_message(self, msg, **tag_kwargs):
        self._add_msg(self.model_chat, ChatRole.SYSTEM, msg, **tag_kwargs)

    def submit_model_chat_and_process_response(self):
        while len(self.model_chat) > self.model_chat_length:
            self._save_chat_state()
            resp = self._get_assistant_response(self._filtered_model_chat)
            self.model_chat_length = len(self.model_chat)
            self.calls_queue += self._parse_model_response(resp)

            while len(self.calls_queue) > 0:
                call = self.calls_queue.pop(0)
                self._add_msg(self.model_chat, ChatRole.ASSISTANT, call,
                              example_num=self.example_num, prompt_iteration=self.prompt_iteration)
                self.model_chat_length += 1
                self._execute_api_call(call)
                self._save_chat_state()

    def _save_chat_state(self):
        self.save_chat_html(self.user_chat, "user_chat.html")
        self.save_chat_html(self.model_chat, "model_chat.html")
        if self.example_num is not None:
            self.save_chat_html(self._filtered_model_chat, f'model_chat_example_{self.example_num}.html')

        chat_dir = os.path.join(self.out_dir, "chat")
        with open(os.path.join(chat_dir, "output_discussion_state.json"), "w") as f:
            json.dump(self.output_discussion_state, f)
        with open(os.path.join(chat_dir, "outputs.json"),
                  "w") as f:
            json.dump(self.outputs, f)
        if self.prompts:
            self.save_prompts_and_config(self.approved_prompts, self.approved_outputs)

    def _parse_model_response(self, resp, max_attempts=2):
        err = ''
        for num_attempt in range(max_attempts):
            if resp.startswith('```python\n'):
                resp = resp[len('```python\n'): -len('\n```')]

            len_resp = len(resp)
            api_indices = sorted(list({
                from_idx + resp[from_idx:].index(name) for name in self.api_names
                for from_idx in range(0, len_resp, len(name)) if name in resp[from_idx:]
            }))
            api_calls = []
            spans = []
            if len(api_indices) > 0:
                for beg, end in zip(api_indices, api_indices[1:] + [len_resp]):
                    last_close_bracket = beg + (resp[beg: end].rfind(')') if ')' in resp[beg: end] else 0) + 1
                    spans.append((beg, last_close_bracket))
                    api_calls.append(resp[beg:last_close_bracket].strip().replace('\n', '\\n').replace('\\n', '  \\n'))

            leftovers = resp
            if len(spans) > 0:
                leftovers = ''.join(
                    [resp[prev[1]: cur[0]] for prev, cur in zip([(0, 0)] + spans, spans + [(len_resp, len_resp)])])
            is_valid = len(api_calls) > 0 and len(leftovers.strip()) == 0

            if is_valid:
                return api_calls

            err += f'\nattempt {num_attempt + 1}: {resp}'
            tmp_chat = self._filtered_model_chat
            self._add_msg(tmp_chat, ChatRole.ASSISTANT, resp)
            self._add_msg(tmp_chat, ChatRole.SYSTEM, self.model_prompts.api_only_instruction)
            resp = self._get_assistant_response(tmp_chat)

        raise ValueError('Invalid model response' + err)

    def _execute_api_call(self, call, max_attempts=2):
        err = ''
        for num_attempt in range(max_attempts):
            try:
                exec(call)
                return
            except SyntaxError as e:
                err += f'\nattempt {num_attempt + 1}: {call}'
                tmp_chat = self._filtered_model_chat
                self._add_msg(tmp_chat, ChatRole.SYSTEM,
                              self.model_prompts.syntax_err_instruction.replace('ERROR', str(e)))
                resp = self._get_assistant_response(tmp_chat)
                call = self._parse_model_response(resp)[0]

        raise ValueError('Invalid call syntax' + err)

    def add_user_message(self, message):
        self._add_msg(self.user_chat, ChatRole.USER, message)
        self.user_chat_length = len(self.user_chat)  # user message is rendered by cpe
        self._add_msg(self.model_chat, ChatRole.USER, message,
                      prompt_iteration=self.prompt_iteration)  # not adding dummy initial user message

    def add_user_message_only_to_user_chat(self, message):
        self._add_msg(self.user_chat, ChatRole.USER, message)
        self.user_chat_length = len(self.user_chat)  # user message is rendered by cpe

    def generate_agent_messages(self):
        self.submit_model_chat_and_process_response()
        agent_messages = []
        if len(self.user_chat) > self.user_chat_length:
            for msg in self.user_chat[self.user_chat_length:]:
                if msg['role'] == ChatRole.ASSISTANT:
                    agent_messages.append(msg)
            self.user_chat_length = len(self.user_chat)

        self._save_chat_state()
        # self.save_prompts_and_config(self.approved_prompts, self.approved_outputs)
        return agent_messages

    def submit_message_to_user(self, message):
        self._add_msg(self.user_chat, ChatRole.ASSISTANT, message)

    def show_original_text(self, example_num):
        txt = self.examples[int(example_num) - 1]
        self._add_msg(chat=self.user_chat, role=ChatRole.ASSISTANT, msg=txt)
        self.add_system_message(f'The original text for Example {example_num} was shown to the user.')

    def task_is_defined(self):
        # open side chat with model
        tmp_chat = self.model_chat[:]
        self._add_msg(tmp_chat, ChatRole.SYSTEM, self.model_prompts.generate_baseline_instruction_task)
        resp = self._get_assistant_response(tmp_chat)
        submit_prmpt_call = self._parse_model_response(resp)[0]
        self.baseline_prompts["model_baseline_prompt"] = submit_prmpt_call[:-2].replace("self.submit_prompt(\"", "")
        logging.info(f"baseline prompt is {self.baseline_prompts['model_baseline_prompt']}")

        self.calls_queue = []
        self.add_system_message(self.model_prompts.analyze_examples)

    def switch_to_example(self, example_num):
        self.model_chat[-1]['example_num'] = None #this is the call to switch to example - we want it to be in the general chat

        example_num = int(example_num)
        self.example_num = example_num
        discuss_ex = self.model_prompts.discuss_example_num.replace('EXAMPLE_NUM', str(self.example_num))
        self.calls_queue = []
        self.add_system_message(discuss_ex, example_num=example_num, prompt_iteration=self.prompt_iteration)

    def submit_prompt(self, prompt):
        prev_discussion_cot = (self.output_discussion_state or {}).get('outputs_discussion_CoT', None)
        self.calls_queue = []
        self.prompts.append(prompt)
        self.model_chat[-1]['prompt_iteration'] = None
        self.model_chat[-1]['example_num'] = None

        futures = {}
        with ThreadPoolExecutor(max_workers=len(self.examples)) as executor:
            for i, example in enumerate(self.examples):
                prompt_str = build_few_shot_prompt(prompt,
                                                   [],  # currently doing zero-shot summarization
                                                   self.target_bam_client.parameters['model_id'])
                prompt_str = prompt_str.format(text=example)
                futures[i] = executor.submit(self._generate_output, prompt_str)

        self.output_discussion_state = {
            'model_outputs': [None] * len(self.examples),
            'user_chat_begin': self.user_chat_length
        }
        self.add_system_message(self.model_prompts.result_intro, prompt_iteration=self.prompt_iteration)
        for i, f in futures.items():
            output = f.result()
            example_num = i + 1
            self.add_system_message(f'Example {example_num}: {output}',
                                    example_num=example_num, prompt_iteration=self.prompt_iteration)
            self.output_discussion_state['model_outputs'][i] = output

        #side chat - generated output of new prompt vs. the outputs that were accepted outputs
        if len(self.prompts) > 1 and prev_discussion_cot is not None:
            tmp_chat = [{'role': ChatRole.SYSTEM, 'content': '\n'.join([
                self.model_prompts.analyze_new_prompt_task,
                self.model_prompts.analyze_new_prompt_accepted_outputs,
                *[f'Example{i + 1}: {o}' for i, o in enumerate(self.outputs)],
                self.model_prompts.analyze_new_prompt_new_outputs,
                *[m['content'] for m in self.model_chat[-len(self.examples):]],
            ])}]

            response = self._get_assistant_response(tmp_chat)
            self.save_chat_html(tmp_chat + [{'role': ChatRole.ASSISTANT, 'content': response}],
                                f'CoT_{self.cot_count}.html')
            self.cot_count += 1
            self.add_system_message(response, prompt_iteration=self.prompt_iteration)

        self.add_system_message(
            self.model_prompts.analyze_result_instruction.replace('NUM_EXAMPLES', str(len(self.examples))),
            prompt_iteration=self.prompt_iteration)

    def output_accepted(self, example_num, output):
        example_idx = int(example_num) - 1
        self.outputs[example_idx] = output
        self.model_chat[-1]['example_num'] = None
        self.model_chat[-1]['prompt_iteration'] = None
        if len(self.calls_queue) == 0:
            if example_idx < len(self.examples) - 1:
                self.calls_queue.append(f'self.switch_to_example({example_idx + 2})')
            else:
                self.calls_queue.append('self.end_outputs_discussion()')

    def end_outputs_discussion(self):
        self.calls_queue = []
        temp_chat = []
        self._add_msg(temp_chat, ChatRole.SYSTEM,
                      self.model_prompts.analyze_discussion_task_begin.replace('PROMPT', self.prompts[-1]))
        temp_chat += self.user_chat[self.output_discussion_state['user_chat_begin']:]
        self._add_msg(temp_chat, ChatRole.SYSTEM, self.model_prompts.analyze_discussion_task_end)
        recommendations = self._get_assistant_response(temp_chat)
        self._add_msg(temp_chat, ChatRole.SYSTEM, recommendations)
        self.save_chat_html(temp_chat, f'CoT_{self.cot_count}.html')
        self.cot_count += 1
        self.output_discussion_state['outputs_discussion_CoT'] = temp_chat

        self.add_system_message(recommendations + '\n' + self.model_prompts.analyze_discussion_continue)

    def conversation_end(self):
        self.prompt_conv_end = True
        self._save_chat_result()
        model_id = self.target_model
        self.few_shot_prompt = build_few_shot_prompt(self.prompts[-1], self.approved_outputs, model_id)
        model_id = self.target_bam_client.parameters['model_id']
        end = self.model_prompts.conversation_end_instruction.replace('TARGET_MODEL', model_id)
        self.add_system_message(end)

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
            self.add_system_message(f'Example {example_num}: {ex}', example_num=example_num)
        self.example_num = None

        self.add_system_message(self.model_prompts.task_definition_instruction)

        self.submit_model_chat_and_process_response()

    def process_examples(self, df, dataset_name):
        self.dataset_name = dataset_name
        self.enable_upload_file = False
        examples = df['text'].tolist()[:3]
        self.init_chat(examples)

    @property
    def result_json_file(self):
        return os.path.join(self.out_dir, 'chat_result.json')

    def _save_chat_result(self):
        data = {
            'examples': self.examples,
            'accepted_outputs': self.outputs,
            'prompts': self.prompts,
            'baseline_prompts': self.baseline_prompts,
            'target_model': self.target_bam_client.parameters['model_id'],
            'dataset_name': self.dataset_name,
            'sent_words_count': self.bam_client.sent_words_count,
            'received_words_count': self.bam_client.received_words_count,
            'config_name': self.config_name
        }
        with open(self.result_json_file, 'w') as f:
            json.dump(data, f)
        if self.prompt_conv_end:
            with open(os.path.join(self.out_dir, "prompt_conv_end.Done"), "w"):
                pass

    @classmethod
    def _read_chat_outputs(self, path):
        model_chat_df = pd.read_csv(os.path.join(f"{path}/chat/", "model_chat.csv"))
        model_chat_list = model_chat_df.to_dict("records")
        model_chat_list = [{k: v for k, v in record.items() if pd.notna(v)} for record in model_chat_list]
        user_chat_df = pd.read_csv(os.path.join(f"{path}/chat/", "user_chat.csv"))
        user_chat_list = user_chat_df.to_dict("records")
        user_chat_list = [{k: v for k, v in record.items() if pd.notna(v)} for record in user_chat_list]

        with open(os.path.join(os.path.join(f"{path}/chat/", "prompts.json")), "r") as f:
            prompts = [x['prompt'] for x in json.load(f)]
        with open(os.path.join(f"{path}/chat/", "config.json"), "r") as f:
            config = json.load(f)
        with open(os.path.join(f"{path}/chat/", "output_discussion_state.json"), "r") as f:
            discussion_state = json.load(f)
        with open(os.path.join(f"{path}/chat/", "outputs.json"), "r") as f:
            outputs = json.load(f)
        return model_chat_list, user_chat_list, prompts, discussion_state, outputs, config

    def load_chat_to_manager(self, path):
        model_chat, user_chat, prompts, discussion_state, outputs, config = self._read_chat_outputs(path)
        dataset_dirs = dataset_name_to_dir[config['dataset']]
        data_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", dataset_dirs["train"]))
        self.process_examples(data_df, config['dataset'])
        self.model_chat = model_chat
        self.model_chat_length = config["model_chat_length"]
        self.user_chat = user_chat
        self.user_chat_length = config["user_chat_length"]
        self.prompts = prompts
        self.output_discussion_state = discussion_state
        self.example_num = config["example_num"]
        self.enable_upload_file = False
        self.outputs = outputs
        return self, config['dataset']
