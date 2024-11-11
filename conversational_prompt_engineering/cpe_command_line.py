import json
import os
import pathlib
import datetime
import logging


from conversational_prompt_engineering.backend.callback_chat_manager import CallbackChatManager

def start_new_chat(output_dir, config_name, chat_model_name, target_model_name, dataset_short_name):
    from conversational_prompt_engineering.configs.config_utils import load_config
    from conversational_prompt_engineering.backend.util.llm_clients.llm_clients_loader import get_client_classes
    from conversational_prompt_engineering.util.csv_file_utils import read_user_csv_file
    from conversational_prompt_engineering.data.dataset_utils import load_dataset_mapping

    config = load_config(config_name)
    chat_llm_client_class = get_client_classes(config.get("General", "chat_llm_api"))
    target_model_llm_client_class = get_client_classes(config.get("General", "target_model_llm_api"))

    manager = CallbackChatManager(model=chat_model_name,
                                  target_model=target_model_name,
                                  chat_llm_client=chat_llm_client_class,
                                  target_model_llm_client=target_model_llm_client_class,
                                  output_dir=output_dir,
                                  config_name=config_name)

    #handle dataset:
    dataset_mapping = load_dataset_mapping(config)
    dataset_files = dataset_mapping[dataset_short_name]
    manager.add_user_message_only_to_user_chat(f"Selected data: {dataset_short_name}")

    manager.process_examples(read_user_csv_file(dataset_files["train"]), dataset_short_name)
    last_printed = print_chat(manager.user_chat)
    while (not manager.prompt_conv_end):
        user_message = input(f'{decorate_role("user")}' + " : ")
        manager.add_user_message(user_message)
        manager.generate_agent_messages()
        last_printed = print_chat(manager.user_chat, last_printed+2) #skip the user's message that already appears
        #print(last_printed)

def decorate_role(role):
    color_codes = {"user": "162;227;252", "assistant": "255;201;227", "system": "245;215;144"}
    if role in color_codes:
        return f"\033[1m\033[48;2;{color_codes[role]}m{role}\033[0m"

def print_chat(user_chat, start_from=0):
    if start_from:
        user_chat = user_chat[start_from:]
    for i, message in enumerate(user_chat):
        print(f"{decorate_role(message['role'])} : {message['content']}")
    return i + start_from


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    output_dir = os.path.join("./_out",  datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"), )
    start_new_chat(output_dir, "production", "llama-3", "llama-3", "Wikipedia Movie pages")