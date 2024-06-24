
from genai.schema import ChatRole


LLAMA_END_OF_MESSAGE = "<|eot_id|>"

LLAMA_START_OF_INPUT = '<|begin_of_text|>'


GRANITE_SYSTEM_MESSAGE = 'You are Granite Chat, an AI language model developed by IBM. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior. You always respond to greetings (for example, hi, hello, g\'day, morning, afternoon, evening, night, what\'s up, nice to meet you, sup, etc) with "Hello! I am Granite Chat, created by IBM. How can I help you today?". Please do not say anything else and do not start a conversation.'

BASELINE_SUMMARIZATION_INSTRUCTION = 'Summarize the following text in 2-3 sentences, highlighting the main ideas and key points.'


def _get_llama_header(role):
    return "<|start_header_id|>" + role + "<|end_header_id|>"


def build_few_shot_prompt(prompt, texts_and_summaries, model_id):
    if 'llama' in model_id:
        return build_few_shot_prompt_llama(prompt, texts_and_summaries)
    elif 'mixtral' in model_id:
        return build_few_shot_prompt_mixtral(prompt, texts_and_summaries)
    elif 'granite' in model_id:
        return build_few_shot_prompt_granite(prompt, texts_and_summaries)
    else:
        raise Exception(f"model {model_id} not supported")


def build_few_shot_prompt_granite(prompt, texts_and_summaries):
    system_prompt = f'<|{ChatRole.SYSTEM}|>\n{GRANITE_SYSTEM_MESSAGE}'
    prompt = system_prompt + '\n' + f'<|{ChatRole.USER}|>' + '\n' + prompt + '\nYour response should only include the answer. Do not provide any further explanation.'
    if len(texts_and_summaries) > 0:
        prompt += "\n\nHere are some examples, complete the last one:\n"
        for item in texts_and_summaries:
            text = item['text']
            summary = item['summary']
            prompt += f"Text:\n{text}\nSummary:\n{summary}\n\n"
    else:
        prompt += "\n\n"
    prompt += 'Text:\n{text}\nSummary:\n' + f'<|{ChatRole.ASSISTANT}|>'
    return prompt


def build_few_shot_prompt_mixtral(prompt, texts_and_summaries):
    prompt = f'[INST] {prompt}\n\n'
    if len(texts_and_summaries) > 0:
        if len(texts_and_summaries) > 1:  # we already have at least two approved summary examples
            prompt += "Here are some typical text examples and their corresponding desired outputs."
        else:
            prompt += "Here is an example of a typical text and its desired output."
        for item in texts_and_summaries:
            text = item['text']
            summary = item['summary']
            prompt += f"\n\nText: {text}\n\nDesired output: {summary}"
        prompt += "\n\nNow, please generate the desired output for the following text.\n\n"
    prompt += "Text: {text}\n\nDesired output: [\INST]"
    return prompt


def build_few_shot_prompt_llama(prompt, texts_and_summaries):
    summary_prefix = "Here is a desired output of the text:"
    prompt = LLAMA_START_OF_INPUT + _get_llama_header(ChatRole.USER) + "\n\n" + prompt + "\n\n"
    if len(texts_and_summaries) > 0:
        if len(texts_and_summaries) > 1:  # we already have at least two approved summary examples
            prompt += "Here are some typical text examples and their corresponding desired outputs."
        else:
            prompt += "Here is an example of a typical text and its desired output."
        for i, item in enumerate(texts_and_summaries):
            if i > 0:
                prompt += _get_llama_header(ChatRole.USER)
            text = item['text']
            summary = item['summary']
            prompt += f"\n\nText: {text}\n\n{LLAMA_END_OF_MESSAGE}" \
                      f"{_get_llama_header(ChatRole.ASSISTANT)}{summary_prefix}{summary}{LLAMA_END_OF_MESSAGE}"
        prompt += _get_llama_header(ChatRole.USER) + "\n\nNow, please generate a desired output to the following text.\n\n"
    prompt += "Text: {text}\n\n" + LLAMA_END_OF_MESSAGE + _get_llama_header(ChatRole.ASSISTANT) + summary_prefix
    return prompt
