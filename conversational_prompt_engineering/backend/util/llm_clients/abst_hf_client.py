# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from conversational_prompt_engineering.backend.util.llm_clients.abst_llm_client import AbstLLMClient, HumanRole
from genai.schema import ChatRole
from conversational_prompt_engineering.backend.prompt_building_util import TargetModelHandler
from conversational_prompt_engineering.backend.util.target_prompt import ConversationPrompt


class AbstHFClient(AbstLLMClient):

    def __init__(self, model_id):
        super(AbstHFClient, self).__init__(model_id)

    def format_chat(self, conversation):
        #no need to format, HF handles it internally
        return conversation

    def _format_as_chat_for_target_model(self, prompt, texts_and_outputs):
        model_short_name = self.parameters['model_short_name']
        model_vars = TargetModelHandler().get_model_vars(model_short_name)
        test_example_prefix = model_vars.get('test_example_prefix', '').strip()
        test_example_placeholder = model_vars.get('test_example_placeholder', '').strip() + "\n\n"
        input_prefix = model_vars.get('input_prefix', '').strip()

        if len(texts_and_outputs) == 0:
            return [{"role": ChatRole.USER,
                     "content": f"{prompt}\n\n{test_example_prefix}\n\n{input_prefix}\n\n{test_example_placeholder}"}]

        if len(texts_and_outputs) > 1:  # we already have at least two approved summary examples
            examples_prefix = "Here are some typical text examples and their corresponding desired outputs."
        else:
            examples_prefix = "Here is an example of a typical text and its desired output."
        user_prompt = f"{prompt}\n\n{examples_prefix}"
        conversation = []
        desired_output_prefix = model_vars.get('text_example_prefix', '').strip()
        for i, item in enumerate(texts_and_outputs):
            if i > 0:
                prompt += model_vars.get('few_shot_examples_prefix', '').strip()
            text = item['text']
            output = item['output']
            user_prompt += f"\n\n{input_prefix} {text}"
            conversation.append({"role": ChatRole.USER, "content": user_prompt})
            conversation.append({"role": ChatRole.ASSISTANT, "content": f"{desired_output_prefix} {output}"})
        conversation.append(
            {"role": ChatRole.USER, "content": f"{test_example_prefix}\n\n{input_prefix} {test_example_placeholder}"})
        return conversation

    #returns a chat structure.
    def format_prompt_for_target_model(self, prompt, texts_and_outputs):
        conversation = self._format_as_chat_for_target_model(prompt=prompt, texts_and_outputs=texts_and_outputs)
        return ConversationPrompt(prompt=conversation)

