import datetime
import hashlib
import logging
import os
import re
from functools import partial

import pandas as pd
from genai.schema import ChatRole

from conversational_prompt_engineering.backend.callback_chat_manager import CallbackChatManager
from conversational_prompt_engineering.backend.chat_manager_util import create_model_client, format_chat
from conversational_prompt_engineering.backend.evaluation_core import Evaluation
from conversational_prompt_engineering.backend.prompt_building_util import build_few_shot_prompt
from conversational_prompt_engineering.pages.evaluation import prompt_types, dimensions
from conversational_prompt_engineering.util.csv_file_utils import read_user_csv_file


def apply_model_template_to_prompt(prompt, model):
    if model == 'llama-3':
        return f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}' \
               f'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'


def process_persona_output(output):
    # persona = re.split('\*\*\*\*\*persona_\d+\*\*\*\*\*\n', output[0])

    persona = re.split('\*\*\*\*\*persona_\d+\*\*\*\*\*\n', output)
    # persona = output.split('*****persona_3*****\n')
    return persona[1:]


def persona_generator(credentials, model, api, task_description, num_of_persona):
    bam_client = create_model_client(credentials=credentials, model_name=model, api=api)

    prompt = f'Given the general task of {task_description}, generate {num_of_persona}' \
             f' persona with different preferences and requirements from the output.' \
             f'\nIn the output, use the following title for each persona: ' \
             f'\'*****persona_i\' where i is the persona index. For example:\n' \
             f'*****persona_1' \
             f'*****persona_2'
    apply_model_template_to_prompt(prompt, model)

    output = bam_client.send_messages(prompt, max_new_tokens=1000)
    print(output[0])
    persona_array = process_persona_output(output[0][0])
    return persona_array


class ModelBasedUser:
    def __init__(self, bam_client, persona, task):
        self.bam_client = bam_client
        self.persona = persona
        self.persona_instruction = {
            'role': ChatRole.SYSTEM,
            'content': f'Below is a conversation between the user and AI assistant. '
                       f'General structure of the conversation is as follows:'
                       f'1. the assistant finds out the type of task, looks at the examples and asks the user few questions to figure out the initial prompt.\n'
                       f'2. the assistant presents the prompt to the user, and if the user agrees to try it uses it to generate the outputs for the text examples.\n'
                       f'3. the assistant presents the generated outputs to the user, and discusses them with him in order to improve them if necessary.\n'
                       f'4. if all the outputs produced by the prompt were accepted by the user without modification, the goal is achieved.\n'
                       f'5. if there were changes to the produced outputs, the assistant suggests a new prompt that takes into account these changes, and returns to the step 2\n'
                       f'In this case the user wants the assistant to build a prompt for the task of {task}. '
                       f'The user persona can be described as follows:\n{persona}\n'
        }

        # this message is shown in UI and is not part of the user chat
        self.user_chat_opening = [{
            'role': ChatRole.ASSISTANT,
            'content': "Hello! I'm an IBM prompt building assistant. "
                       "In the following session we will work together through a natural conversation, "
                       "to build an effective instruction – a.k.a. prompt – personalized for your task and data."
        }]

    def turn(self, assistant):
        # relevant example texts
        if assistant.example_num is None:
            example_txt = 'The text examples under discussion are:\n'
            example_txt += '\n'.join([f'Example {i + 1}:\n{ex}' for i, ex in enumerate(assistant.examples)])
        else:
            txt = assistant.examples[assistant.example_num - 1]
            example_txt = f'The example under discussion is:\nExample {assistant.example_num}:\n{txt}'
        example_msg = {'role': ChatRole.SYSTEM, 'content': example_txt}

        # compose the conversation
        chat = [self.persona_instruction, example_msg] + self.user_chat_opening + assistant.user_chat

        chat.append({
            'role': ChatRole.SYSTEM,
            'content': 'You are the user persona. Answer to the assistant. Be brief.'
        })

        formatted_chat = format_chat(chat, self.bam_client.parameters['model_id'])
        generated_texts, stats_dict = self.bam_client.send_messages(formatted_chat)
        return '\n'.join(generated_texts)

    def select_best_worst(self, assistant, example_text, output_options):
        def _parse_num(txt, prefix):
            return int(txt[txt.index(prefix) + len(prefix):].split(' ')[0])

        chat = [self.persona_instruction] + self.user_chat_opening + assistant.user_chat
        chat += [
            {
                'role': ChatRole.SYSTEM,
                'content':
                    'The conversation has finished. Now the user persona is asked to evaluate the result:\n'
                    'For the following text and the presented output options the user has to choose the best and the worst option. '
                    'The response format must be like this: best=<option_number> worst=<option_number>\n'
                    'You are the the user persona. Choose the best and the worst output options.'
            },
            {
                'role': ChatRole.SYSTEM,
                'content': f'Text:\n{example_text}\n' + '\n'.join(output_options)
            },
        ]

        formatted_chat = format_chat(chat, self.bam_client.parameters['model_id'])
        generated_texts, stats_dict = self.bam_client.send_messages(formatted_chat)
        out_txt = '/n'.join(generated_texts)
        best_idx = _parse_num(out_txt, 'best=') - 1
        worst_idx = _parse_num(out_txt, 'worst=') - 1
        return best_idx, worst_idx


class AutoChat:
    def __init__(self, email_address, credentials, model, api, persona, task, ds_name, train_csv, eval_csv) -> None:
        self.eval_csv = eval_csv

        sha1 = hashlib.sha1()
        sha1.update(credentials["key"].encode('utf-8'))
        conv_id = sha1.hexdigest()[:16]  # deterministic hash of 16 characters

        user_dir = email_address.split("@")[0]  # default is self.conv_id
        user_time_dir = f'_out/{user_dir}/{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}'

        # prepare the assistant
        assistant_out_dir = user_time_dir
        os.makedirs(assistant_out_dir, exist_ok=True)
        self.assistant_manager = CallbackChatManager(credentials=credentials,
                                                     model=model,
                                                     target_model=model,
                                                     conv_id=conv_id,
                                                     api=api,
                                                     email_address=email_address,
                                                     output_dir=assistant_out_dir,
                                                     config_name='main')
        train_df = read_user_csv_file(train_csv)
        self.assistant_manager.process_examples(train_df, ds_name)
        examples = self.assistant_manager.examples

        # prepare the user
        user_out_dir = os.path.join(user_time_dir, 'user')
        os.makedirs(user_out_dir, exist_ok=True)
        self.user_manager = ModelBasedUser(bam_client=self.assistant_manager.bam_client,
                                           persona=persona, task=task)

    def create_prompt(self):
        # roll the chat
        while not self.assistant_manager.prompt_conv_end:
            user_msg = self.user_manager.turn(self.assistant_manager)
            self.assistant_manager.add_user_message(user_msg)
            self.assistant_manager.generate_agent_messages()

    def evaluate(self):
        def _build_prompt_fn(prompt, few_shot):
            model_id = self.assistant_manager.target_bam_client.parameters['model_id']
            return partial(build_few_shot_prompt, prompt=prompt, texts_and_summaries=few_shot, model_id=model_id)

        prompt_type_metadata = {
            "baseline": {
                "title": "Prompt 1 (Baseline prompt)",
                "build_func": _build_prompt_fn("Summarize this text", [])
            },
            "zero_shot": {
                "title": "Prompt 2 (CPE zero shot prompt)",
                "build_func": _build_prompt_fn(self.assistant_manager.approved_prompts[-1]['prompt'], [])
            },
            "few_shot": {
                "title": "Prompt 3 (CPE few shot prompt)",
                "build_func": _build_prompt_fn(self.assistant_manager.approved_prompts[-1]['prompt'],
                                               self.assistant_manager.approved_outputs)
            }
        }
        eval_texts = read_user_csv_file(self.eval_csv).text.tolist()
        eval_prompts = [prompt_type_metadata[t]["build_func"]() for t in prompt_types]

        evaluation = Evaluation(self.assistant_manager.target_bam_client)
        generated_data = evaluation.summarize(eval_prompts, prompt_types, eval_texts)

        dim = dimensions[0]
        for example in generated_data:
            idx2prompt = sorted(example['mixed_indices_mapping_to_prompt_type'].items())
            outputs = [f'\nOutput {i + 1}:\n{example[p_type + "_output"]}' for i, p_type in idx2prompt]
            best_idx, worst_idx = self.user_manager.select_best_worst(self.assistant_manager, example['text'], outputs)

            example[f'sides_{(dim, "Best")}'] = best_idx
            example[f'ranked_prompt_{(dim, "Best")}'] = idx2prompt[best_idx][1]
            example[f'sides_{(dim, "Worst")}'] = worst_idx
            example[f'ranked_prompt_{(dim, "Worst")}'] = idx2prompt[worst_idx][1]

        out_dir = os.path.join(self.assistant_manager.out_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(sorted(generated_data, key=lambda x: x["index"])).drop(columns=['index'])
        eval_csv = os.path.join(out_dir, "eval_results.csv")
        df.to_csv(eval_csv, index=False)
        logging.info(f'Evaluation results saved to {os.path.abspath(eval_csv)}')


def run_auto_chat(email_address, credentials, model, api):
    persona = """
        Name: Bob
        Occupation: Data scientist and prompt engineer
        Goal: Create a prompt that would summarize legal texts in short summaries easily read and understood by everyone.     
        Requirements: 
        - The summary should written in plain form, without formatting
        - The summary should use simple language understood by everyone 
        - The summary should cover all the important points of the orig text
        - The summary should be as short as possible, but still covering all the important points
        - The prompt should not be wordy, with a lot of special cases. It should be generic, compact, and straight to the point.
    """
    chat = AutoChat(
        email_address=email_address, credentials=credentials, model=model, api=api,
        persona=persona, task='text_summarization',
        ds_name='movie reviews',
        train_csv='data/public/legal_plain_english/train.csv',
        eval_csv='data/public/legal_plain_english/eval.csv',
    )
    chat.create_prompt()
    chat.evaluate()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    common_param = {
        'credentials': {'key': os.environ.get('BAM_APIKEY')},
        'model': 'llama-3',
        'api': 'bam',
    }

    # persona_generator(**common_param, task_description='text summarization', num_of_persona=3)

    run_auto_chat(**common_param, email_address=os.environ.get('IBM_EMAIL'))
