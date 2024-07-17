import datetime
import hashlib
import logging
import os
import re

from genai.schema import ChatRole

from conversational_prompt_engineering.backend.callback_chat_manager import CallbackChatManager
from conversational_prompt_engineering.backend.chat_manager_util import create_model_client, format_chat
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
    def __init__(self, bam_client, persona, task, examples):
        self.bam_client = bam_client
        self.persona = persona
        instruction = f'You you need to act as a persona who is interested in the general task of {task}.' \
                      f'Your persona is described as follows:\n+{self.persona}\n' \
                      f'You are about to chat with a system that helps users build a personalized instruction - a.k.a. prompt - for their specific task and data.' \
                      f'You should work together with the system through a natural conversation, to build your own prompt. Please make sure to express the specific expectation and requirements of your persona from the task output.'

        instruction += '\nHere are the text examples you will work on:\n' + \
                       '\n'.join([f'Example {i + 1}:\n{ex}' for i, ex in enumerate(examples)])

        static_welcome_msg = \
            "Hello! I'm an IBM prompt building assistant. In the following session we will work together through a natural conversation, to build an effective instruction – a.k.a. prompt – personalized for your task and data."

        self.system_instruction = [
            {'role': ChatRole.SYSTEM, 'content': instruction},
            {'role': ChatRole.ASSISTANT, 'content': static_welcome_msg},
        ]

    def turn(self, user_chat):
        conversation = format_chat(self.system_instruction + user_chat, self.bam_client.parameters['model_id'])
        generated_texts, stats_dict = self.bam_client.send_messages(conversation)
        return '\n'.join(generated_texts)


class AutoChat:
    def __init__(self, email_address, credentials, model, api, persona, task, examples_csv, ds_name) -> None:
        sha1 = hashlib.sha1()
        sha1.update(credentials["key"].encode('utf-8'))
        conv_id = sha1.hexdigest()[:16]  # deterministic hash of 16 characters

        user_dir = email_address.split("@")[0]  # default is self.conv_id
        user_time_dir = f'_out/{user_dir}/{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}'

        # prepare the assistant
        assistant_out_dir = os.path.join(user_time_dir, 'assistant')
        os.makedirs(assistant_out_dir, exist_ok=True)
        self.assistant_manager = CallbackChatManager(credentials=credentials,
                                                     model=model,
                                                     target_model=model,
                                                     conv_id=conv_id,
                                                     api=api,
                                                     email_address=email_address,
                                                     output_dir=assistant_out_dir,
                                                     config_name='main')
        examples_df = read_user_csv_file(examples_csv)
        self.assistant_manager.process_examples(examples_df, ds_name)
        examples = self.assistant_manager.examples

        # prepare the user
        user_out_dir = os.path.join(user_time_dir, 'user')
        os.makedirs(user_out_dir, exist_ok=True)
        self.user_manager = ModelBasedUser(bam_client=self.assistant_manager.bam_client,
                                           persona=persona, task=task,
                                           examples=examples)

    def go(self):
        # roll the chat
        while not self.assistant_manager.prompt_conv_end:
            user_msg = self.user_manager.turn(self.assistant_manager.user_chat)
            self.assistant_manager.add_user_message(user_msg)
            self.assistant_manager.generate_agent_messages()

        print("Done")


def run_auto_chat(email_address, credentials, model, api):
    persona = """
        Name: Busy Executive
        Age: 45
        Occupation: CEO of a large corporation
        Goal: To quickly grasp the main points of a long document or article
        Requirements: 
        - The summary should be concise (less than 100 words)
        - The summary should highlight the key takeaways and main points
        - The summary should be easy to read and understand
    """
    chat = AutoChat(
        email_address=email_address,
        credentials=credentials,
        model=model,
        api=api,
        persona=persona,
        task='text_summarization',
        examples_csv='data/public/movie reviews/train.csv',
        ds_name='movie reviews'
    )
    chat.go()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    common_param = {
        'credentials': {'key': os.environ.get('BAM_APIKEY')},
        'model': 'llama-3',
        'api': 'bam',
    }

    # persona_generator(**common_param, task_description='text summarization', num_of_persona=3)

    run_auto_chat(**common_param, email_address=os.environ.get('IBM_EMAIL'))
