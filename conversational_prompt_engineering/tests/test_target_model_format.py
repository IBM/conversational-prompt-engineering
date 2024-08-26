from conversational_prompt_engineering.backend.prompt_building_util import TargetModelHandler

target_model = "meta-llama/llama-3-70b-instruct"
prompt = "summarize this text."
fs_examples = []  # [{'text': 'text_1', 'output': 'output'}]
formatted_prompt = TargetModelHandler().format_prompt(model=target_model, prompt=prompt, texts_and_outputs=fs_examples)
print(formatted_prompt)