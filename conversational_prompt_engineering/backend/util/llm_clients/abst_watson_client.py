# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from conversational_prompt_engineering.backend.util.llm_clients.abst_llm_client import AbstLLMClient, HumanRole
from conversational_prompt_engineering.backend.prompt_building_util import LLAMA_END_OF_MESSAGE, \
    _get_llama_header, LLAMA_START_OF_INPUT
from genai.schema import ChatRole
from conversational_prompt_engineering.backend.prompt_building_util import TargetModelHandler
from conversational_prompt_engineering.backend.util.target_prompt import StringPrompt

def format_chat(chat, short_model_name):
    if any([name in short_model_name for name in ['mixtral', 'prometheus']]):
        bos_token = '<s>'
        eos_token = '</s>'
        chat_for_mixtral = []
        prev_role = None
        for m in chat:
            if m["role"] == prev_role:
                chat_for_mixtral[-1]["content"] += "\n" + m["content"]
            else:
                chat_for_mixtral.append(m)
            prev_role = m["role"]

        for m in chat_for_mixtral:
            if m["role"] == 'user':
                m["content"] = 'user: ' + m["content"]
            elif m["role"] == 'system':
                m["role"] = 'user'
                m["content"] = 'system: ' + m["content"]

        prompt = bos_token
        for m in chat_for_mixtral:
            if m['role'] == 'user':
                prompt += '[INST] ' + m['content'] + ' [/INST] '
            else:
                prompt += m['content'] + eos_token + ' '
        return prompt
    elif 'llama' in short_model_name:
        msg_str = LLAMA_START_OF_INPUT
        for m in chat:
            msg_str += _get_llama_header(m['role']) + "\n\n" + m['content'] + LLAMA_END_OF_MESSAGE
        msg_str += _get_llama_header(ChatRole.ASSISTANT)
        return msg_str
    else:
        raise ValueError(f"model {model_id} not supported")


class AbstWatsonClient(AbstLLMClient):
    def __init__(self, model_id):
        super(AbstWatsonClient, self).__init__(model_id)

    def format_chat(self, conversation):
        return format_chat(conversation, self.parameters['model_short_name'])

    # returns a single string
    def format_prompt_for_target_model(self, prompt, texts_and_outputs):
        formatted_str = TargetModelHandler().format_prompt(model_short_name=self.parameters['model_short_name'],
                                                  prompt=prompt, texts_and_outputs=texts_and_outputs)
        return StringPrompt(formatted_str)

