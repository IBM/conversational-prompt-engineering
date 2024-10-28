# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import logging
import time
import os
import json

import pandas as pd

from conversational_prompt_engineering.backend.prompt_building_util import TargetModelHandler
from genai.schema import ChatRole


def extract_delimited_text(txt, delims):
    try:
        if type(delims) is str:
            delims = [delims]
        for delim in delims:
            if delim in txt:
                begin = txt.index(delim) + len(delim)
                end = begin + txt[begin:].index(delim)
                return txt[begin:end]
        return txt  # delims not found in text
    except ValueError:
        return txt


def create_model_client(model_name, llm_client):
    with open(os.path.join(os.path.dirname(__file__),"model_params.json"), "r") as f:
        params = json.load(f)
    model_params = {x: y for x, y in params['models'][model_name].items()}
    model_params["model_short_name"] = model_name
    model_params["llm_client"] = llm_client.__name__
    endpoint = params["endpoints"].get(llm_client.__name__) #end point is not always needed, so we allow it to be None
    try:
        return llm_client(endpoint, model_params)
    except Exception as e:
        raise ValueError(f'Error generating model client: {e.error_msg}')




class ChatManagerBase:
    def __init__(self, model, target_model, chat_llm_client, target_model_llm_client, output_dir, config_name) -> None:
        logging.info(f"selected {model}")
        logging.info(f"selected target {target_model}")

        self.llm_client = create_model_client(model, chat_llm_client)
        self.target_llm_client = create_model_client(target_model, target_model_llm_client)
        self.dataset_name = None
        self.state = None
        self.timing_report = []
        self.out_dir = output_dir
        self.config_name = config_name
        logging.info(f"output is saved to {os.path.abspath(self.out_dir)}")


    def save_config(self):
        chat_dir = os.path.join(self.out_dir, "chat")
        os.makedirs(chat_dir, exist_ok=True)
        with open(os.path.join(chat_dir, "config.json"), "w") as f:
            json.dump({"model_params": self.llm_client.parameters,
                       "dataset": self.dataset_name,
                       "config_name": self.config_name,
                       "target_model_params": self.target_llm_client.parameters}, f)

    def save_chat_html(self, chat, file_name):
        def _format(msg):
            role = msg['role'].upper()
            txt = msg['content']
            relevant_tags = {k: msg[k] for k in (msg.keys() - {'role', 'content', 'tooltip'})}
            tags = ""
            if relevant_tags:
                tags = str(relevant_tags)
            return f"<p><b>{role}: </b>{txt} {tags}</p>".replace("\n", "<br>")

        chat_dir = os.path.join(self.out_dir, "chat")
        os.makedirs(chat_dir, exist_ok=True)
        df = pd.DataFrame(chat)
        df.to_csv(os.path.join(chat_dir, f"{file_name.split('.')[0]}.csv"), index=False)
        with open(os.path.join(chat_dir, file_name), "w") as html_out:
            content = "\n".join([_format(x) for x in chat])
            header = "<h1>IBM Research Conversational Prompt Engineering</h1>"
            html_template = f'<!DOCTYPE html><html>\n<head>\n<title>CPE</title>\n</head>\n<body style="font-size:20px;">{header}\n{content}\n</body>\n</html>'
            html_out.write(html_template)

    def _add_msg(self, chat, role, msg):
        chat.append({'role': role, 'content': msg})

    def print_timing_report(self):
        df = pd.DataFrame(self.timing_report)
        logging.info(df)
        logging.info(f"Average processing time: {df['total_time'].mean()}")
        self.timing_report = sorted(self.timing_report, key=lambda row: row['total_time'])
        logging.info(f"Highest processing time: {self.timing_report[-1]}")
        logging.info(f"Lowest processing time: {self.timing_report[0]}")

    def _generate_output_and_log_stats(self, conversation, client, max_new_tokens=None):
        start_time = time
        generated_texts, stats_dict = client.send_messages(conversation, max_new_tokens)
        elapsed_time = time.time() - start_time.time()
        timing_dict = {"total_time": elapsed_time, "start_time": start_time.strftime("%d-%m-%Y %H:%M:%S")}
        timing_dict.update(stats_dict)
        logging.info(timing_dict)
        self.timing_report.append(timing_dict)
        return generated_texts

    def _generate_output(self, prompt, client=None):
        if client is None:
            client = self.target_llm_client
        generated_texts = self._generate_output_and_log_stats(prompt, client=client)
        agent_response = generated_texts[0]
        logging.info(f"got response from model: {agent_response}")
        return agent_response.strip()

    def _get_assistant_response(self, chat, max_new_tokens=None):
        conversation = self.llm_client.format_chat(chat)
        generated_texts = self._generate_output_and_log_stats(conversation, client=self.llm_client,
                                                              max_new_tokens=max_new_tokens)
        agent_response = ''
        for txt in generated_texts:
            if any([f'<|{r}|>' in txt for r in [ChatRole.SYSTEM, ChatRole.USER]]):
                agent_response += txt[: txt.index('<|')]
                break
            agent_response += txt
        logging.info(f"got response from model: {agent_response}")
        return agent_response.strip()
