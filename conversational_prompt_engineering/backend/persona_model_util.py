import json
import logging
import re



from conversational_prompt_engineering.util.bam import BamGenerate

def apply_model_template_to_prompt(prompt, model):
    if model == 'llama-3':
        return f'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}' \
               f'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n'


def process_persona_output(output):
    # persona = re.split('\*\*\*\*\*persona_\d+\*\*\*\*\*\n', output[0])

    persona = re.split('\*\*\*\*\*persona_\d+\*\*\*\*\*\n',output)
    # persona = output.split('*****persona_3*****\n')
    return persona[1:]



def persona_generator(bam_client, task_description, num_of_persona):

    prompt = f'Given the general task of {task_description}, generate {num_of_persona}' \
             f' persona with different preferences and requirements from the output.' \
             f'\nIn the output, use the following title for each persona: ' \
             f'\'*****persona_i\' where i is the persona index. For example:\n' \
             f'*****persona_1' \
             f'*****persona_2'
    apply_model_template_to_prompt(prompt, model)

    output = bam_client.send_messages(prompt, max_new_tokens=1000)
    print(output[0])
    persona_array = process_persona_output(output[0])
    return persona_array

class ModelBasedUser:
    def __init__(self, bam_client, persona) -> None:
        self.bam_client = bam_client
        self.persona = persona
        self.system_prompt = f'You you need to act as a persona who is interested in the general task of {task_description}.' \
                             f'Your persona is described as follows:\n+{self.persona}\n' \
                             f'You are about to chat with a system that helps users build a personalized instruction - a.k.a. prompt - for their specific task and data.' \
                             f'You should work together with the system through a natural conversation, to build your own prompt. Please make sure to express the specific expectation and requirements of your persona from the task output.'



if __name__ == '__main__':
    model = 'llama-3'
    conv_id = 3
    with open("backend/bam_params.json", "r") as f:
        params = json.load(f)
    logging.info(f"selected {model}")
    logging.info(f"conv id: {conv_id}")
    bam_params = params['models'][model]
    bam_params['api_key'] = 'pak-OU5K14KHKT2O_VIYeVh-8kKqB8dmx0EPy_g1VbG8vfQ'
    bam_params['api_endpoint'] = params['api_endpoint']
    bam_client = BamGenerate(bam_params)
    print(len(persona_generator(bam_client, 'text summarization', 3)))