import os

import pandas as pd
from tqdm import tqdm

from conversational_prompt_engineering.util.bam import Chat

system_text = """You are an assistant that builds an instruction for a text summarization task. Please ask me questions and use my answers to build an initial instruction for this task:

Would you like the summary to be more extractive or abstractive? 

Would you like the summary to be formal or informal? 

How long would you like the summary to be? 

Please do not provide any explanations, unless asked. 

Please ask me the questions one at a time.

In response to a greeting (Hi, hello, good day, etc.) please reply by saying 'Hi! I am your conversational prompt engineering assistant. I will ask you a few questions to better understand your expectations from generated summaries for your use-case.'

When all questions have been answered, please write a message saying 'I have all the necessary information to build an initial instruction for your text summarization task.'.
"""

BAM_END_POINT = "https://bam-api.res.ibm.com"


def test_chat():
    params = {'model_id': 'mistralai/mixtral-8x7b-instruct-v0-1', 'api_key': os.getenv('BAM_API_KEY'),
              'api_endpoint': BAM_END_POINT, 'system_prompt': system_text}
    chat = Chat(params)
    conv_id = chat.conversation_id
    while True:
        user_input = input("Enter input:")
        response, conv_id = chat.send_message(conv_id=conv_id, text=user_input)
        print(f"AI: {response}")


def test_infer():
    params = {'model_id': 'mistralai/mixtral-8x7b-instruct-v0-1', 'api_key': os.getenv('BAM_API_KEY'),
              'api_endpoint': BAM_END_POINT, 'system_prompt': ""}
    chat = Chat(params=params)
    instruction = "Create an informal, abstractive summary that is approximately 20% the length of the original text."
    test_df = pd.read_csv("data/legal_plain_english/test.csv")
    texts = test_df.text.tolist()
    prompts = ["\n\n".join([instruction, "Text: " + text, "Summary: "]) for text in texts]
    generated_summaries = []
    for text in tqdm(prompts):
        gen_summary, _ = chat.send_message(conv_id=None, text=text)
        generated_summaries.append(gen_summary)
    pd.DataFrame({'text': texts, 'generated_summaries': generated_summaries}).to_csv("chat_output.csv", index=False)


if __name__ == "__main__":
    test_chat()

