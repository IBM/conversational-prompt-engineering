import logging
import os
import datetime
import sys
import configparser
import importlib

import pandas as pd
import streamlit as st
import hashlib
from enum import Enum
from genai.schema import ChatRole
from streamlit_js_eval import streamlit_js_eval

from conversational_prompt_engineering.backend.callback_chat_manager import CallbackChatManager
from conversational_prompt_engineering.backend.double_chat_manager import DoubleChatManager
from conversational_prompt_engineering.backend.manager import Manager, Mode
from conversational_prompt_engineering.util.csv_file_utils import read_user_csv_file
from conversational_prompt_engineering.util.upload_csv_or_choose_dataset_component import \
    create_choose_dataset_component_train,add_evaluator_input
from configs.config_names import load_config
from conversational_prompt_engineering.data.dataset_utils import load_dataset_mapping

from st_pages import Page, show_pages, hide_pages

version = "callback manager v1.0.7"
st.set_page_config(layout="wide", menu_items={"About": f"CPE version: {version}"})

show_pages(
    [
        Page("cpe.py", "Chat", ""),
        Page("pages/survey.py", "Survey", ""),
        Page("pages/evaluation.py", "Evaluate", ""),
    ]
)


MUST_HAVE_UPLOADED_DATA_TO_START = True
USE_ONLY_LLAMA = True


class APIName(Enum):
    BAM, Watsonx = "bam", "watsonx"
    def __eq__(self, other):
        if type(self).__qualname__ != type(other).__qualname__:
            return NotImplemented
        return self.name == other.name and self.value == other.value

    def __hash__(self):
        return hash((type(self).__qualname__, self.name))



def old_reset_chat():
    st.session_state.manager = Manager(st.session_state.mode, st.session_state.key)
    st.session_state.messages = []


def reset_chat():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


def new_cycle():
    # 1. create the manager if necessary
    if "manager" not in st.session_state:
        sha1 = hashlib.sha1()
        sha1.update(st.session_state.key.encode('utf-8'))
        st.session_state.conv_id = sha1.hexdigest()[:16]  # deterministic hash of 16 characters
        st.session_state.manager = DoubleChatManager(bam_api_key=st.session_state.key, model=st.session_state.model,
                                                     conv_id=st.session_state.conv_id)
    manager = st.session_state.manager

    # 2. hide evaluation option in sidebar
    # prompts = manager.get_prompts()

    # if len(prompts) < 2:
    #     hide_pages(["Evaluate"])

    # 3. layout reset and upload buttons in 3 columns
    if st.button("Reset chat"):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    create_choose_dataset_component_train(st=st, manager=manager)

    # 4. user input
    if user_msg := st.chat_input("Write your message here"):
        manager.add_user_message(user_msg)

    # 5. render the existing messages
    for msg in manager.user_chat:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # 6. generate and render the agent response
    msg = manager.generate_agent_message()
    if msg is not None:
        with st.chat_message(msg['role']):
            st.write(msg['content'])


def set_output_dir():
    subfolder = st.session_state.email_address.split("@")[0]  # default is self.conv_id
    out_folder = f'_out/{subfolder}/{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}'
    os.makedirs(out_folder, exist_ok=True)
    return out_folder


def callback_cycle():
    # create the manager if necessary
    if "manager" not in st.session_state:
        sha1 = hashlib.sha1()
        sha1.update(st.session_state.credentials["key"].encode('utf-8'))
        st.session_state.conv_id = sha1.hexdigest()[:16]  # deterministic hash of 16 characters

        output_dir = set_output_dir()
        file_handler = logging.FileHandler(os.path.join(output_dir, "out.log"))
        logger = logging.getLogger()
        logger.addHandler(file_handler)

        st.session_state.manager = CallbackChatManager(credentials=st.session_state.credentials,
                                                       model=st.session_state.model,
                                                       target_model=st.session_state.target_model,
                                                       conv_id=st.session_state.conv_id, api=st.session_state.API.value,
                                                       email_address=st.session_state.email_address,
                                                       output_dir=output_dir,
                                                       config_name=st.session_state["config_name"])

    manager = st.session_state.manager

    # layout reset and upload buttons in 3 columns
    if st.button("Reset chat"):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    static_welcome_msg = \
        "Hello! I'm an IBM prompt building assistant. In the following session we will work together through a natural conversation, to build an effective instruction – a.k.a. prompt – personalized for your task and data."

    with st.chat_message(ChatRole.ASSISTANT):
        st.write(static_welcome_msg)

    add_evaluator_input(st)

    uploaded_file = create_choose_dataset_component_train(st=st, manager=manager)
    if uploaded_file:
        manager.add_user_message_only_to_user_chat("Selected data")

    static_upload_data_msg = "To begin, please upload your data, or select a dataset from our datasets catalog above."
    with st.chat_message(ChatRole.ASSISTANT):
        st.write(static_upload_data_msg)

    dataset_is_selected = "selected_dataset" in st.session_state or "csv_file_train" in st.session_state
    if not MUST_HAVE_UPLOADED_DATA_TO_START or dataset_is_selected:
        if user_msg := st.chat_input("Write your message here"):
            manager.add_user_message(user_msg)

        for msg in manager.user_chat[:manager.user_chat_length]:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        # generate and render the agent response
        with st.spinner("Thinking..."):
            if uploaded_file:
                manager.process_examples(read_user_csv_file(st.session_state["csv_file_train"]), st.session_state[
                    "selected_dataset"] if "selected_dataset" in st.session_state else "user")
            messages = manager.generate_agent_messages()
            for msg in messages:
                with st.chat_message(msg['role']):
                    st.write(msg['content'])

    if manager.few_shot_prompt is not None:
        btn = st.download_button(
            label="Download few shot prompt",
            data=manager.few_shot_prompt,
            file_name='few_shot_prompt.txt',
            mime="text"
        )


def old_cycle():
    def show_and_call(prompt, show_message=True):
        st.session_state.messages.append({"role": "user", "content": prompt, "show": show_message})
        if show_message:
            with st.chat_message("user"):
                st.markdown(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.manager.call(
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
            )
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response, "show": True})

    if "key" in st.session_state:
        if st.button("Reset chat"):
            reset_chat()

        mode = st.radio(label="Mode", options=["Basic", "Advanced"],
                        captions=["basic zero-shot -> few-shot (default)",
                                  "basic zero-shot -> custom zero-shot -> few-shot"])

        if "mode" not in st.session_state:
            st.session_state.mode = Mode.Basic

        old_mode = st.session_state.mode
        if mode == "Basic":
            st.session_state.mode = Mode.Basic
        else:
            st.session_state.mode = Mode.Advanced
        new_mode = st.session_state.mode
        if old_mode != new_mode:
            old_reset_chat()

        if "manager" not in st.session_state:
            st.session_state.manager = Manager(st.session_state.mode, st.session_state.key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            if message["show"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        if 'messages' in st.session_state and len(st.session_state.messages) == 0:
            show_and_call(f"hi", show_message=False)  # {threading.get_ident()}
        # st.write("Hi, please provide your BAM API key")
        if prompt := st.chat_input("What is up?"):
            show_and_call(prompt)


def submit_button_clicked(target_model):
    def get_secret_key(env_var_name, text_area_key):
        return getattr(st.session_state, text_area_key) if env_var_name not in os.environ else os.environ[env_var_name]

    # verify credentials
    st.session_state.cred_error = ""
    st.session_state.email_error = ""
    creds_are_ok = False
    if st.session_state.API == APIName.BAM:
        api_key = get_secret_key("BAM_APIKEY","bam_api_key")
        if api_key != "":
            st.session_state.credentials = {'key': api_key}
            creds_are_ok = True
    elif st.session_state.API == APIName.Watsonx:
        api_key = get_secret_key("WATSONX_APIKEY","watsonx_api_key")
        project_id = get_secret_key("PROJECT_ID","project_id")
        if api_key != "" and project_id != "":
            st.session_state.credentials = {'key': api_key,
                                                'project_id': project_id}
            creds_are_ok = True

    if creds_are_ok:
        st.session_state.model = 'llama-3'
        st.session_state.target_model = target_model
    else:
        st.session_state.cred_error = ':heavy_exclamation_mark: Please provide your credentials'

    # verify email:
    if verify_email(st.session_state.email_address_input):  # check text area
        st.session_state.email_address = st.session_state.email_address_input
    else:
        st.session_state.email_error = ':heavy_exclamation_mark: Please provide your email address'


def verify_email(email_address):
    return "@" in email_address and "ibm" in email_address and email_address.index("@") != 0



eval_instructions_for_user = \
    "Welcome to IBM Research Conversational Prompt Engineering (CPE) service.\n\n" \
    "This service is intended to help users build an effective prompt, personalized to their specific summarization use case, through a simple chat with an LLM.\n\n" \
    f"The *prompts* built by CPE are comprised of two parts: an instruction, describing to the LLM in natural language how to generate the summaries; and up to 3 text-summary pairs, exemplifying how summaries should look like.\n\n" \
    "To use and evaluate CPE, proceed according to the following steps:\n\n" \
    "1.	Select a summarization dataset from our catalog. " \
    "Please select the dataset that is most related to your daily work, or if none exists, select the dataset which interests you most. \n\n" \
    "2.	Dedicate a few moments to consider your preferences for generating a summary. " \
    "It may be helpful to download the dataset and go over a few text inputs in order to obtain a better understanding of the task. \n\n" \
    "3.	If you have an instruction you are already working with, you can share it in the top of the page when asked. \n\n" \
    "4.	Follow the chat with the system. \n\n" \
    "5.	Once the system notifies you that the final prompt is ready, please click on the Survey tab to answer a few short questions about your interaction.\n\n" \
    "6.	Finally, click on the Evaluate tab. In this stage we ask you to compare between summaries generated by 3 prompts: " \
    "one comprised of a generic summarization instruction, and two built with the CPE system, with and without text-summary examples. \n\n" \
    "Stages 1-4 typically takes around 15 minutes. Please complete these stages in a single session without interruption, if possible.\n\n" \
    "Generating the summaries for stage 5 could take several minutes, so this stage can be done at a later time.\n\n" \
    "Do not include PII or confidential information in your responses, nor in the data you share.\n\n" \
    "Thank you for your time!"






def load_environment_variables():
    if "API" not in st.session_state: #do it only once
        if "BAM_APIKEY" in os.environ:
            st.session_state.credentials = {}
            st.session_state.API = APIName.BAM
            st.session_state.credentials["key"] = os.environ["BAM_APIKEY"]
        elif "WATSONX_APIKEY" in os.environ:
            st.session_state.credentials = {}
            st.session_state.credentials = {"project_id": os.environ["PROJECT_ID"]}
            st.session_state.API = APIName.Watsonx
            st.session_state.credentials["key"] = os.environ["WATSONX_APIKEY"]
            logging.info(f"credentials from environment variables: {st.session_state.credentials}")
        else:
            st.session_state.API = APIName.BAM
            st.session_state["credentials"] = {}

    if "IBM_EMAIL" in os.environ and verify_email(os.environ["IBM_EMAIL"]):
        st.session_state.email_address = os.environ["IBM_EMAIL"]


def set_credentials():
    def handle_secret_key(cred_key, env_var_name, text_area_key, text_area_label):
        is_disabled = False
        val = "" if not hasattr(st.session_state, text_area_key) else getattr(st.session_state, text_area_key)
        if env_var_name in os.environ:
            val = "****" #hidden
            is_disabled = True
        st.text_input(label=text_area_label, key=text_area_key, disabled=is_disabled, value=val)


    #st.session_state.API = APIName.Watsonx if api == "Watsonx" else APIName.BAM
    if st.session_state.API == APIName.Watsonx:
        handle_secret_key(cred_key='key', env_var_name='WATSONX_APIKEY', text_area_key="watsonx_api_key", text_area_label="Watsonx API key")
        handle_secret_key(cred_key='project_id', env_var_name='PROJECT_ID',text_area_key="project_id", text_area_label="project ID" )
    else:
        handle_secret_key(cred_key='key', env_var_name='BAM_APIKEY',text_area_key="bam_api_key", text_area_label="BAM API key" )

    if hasattr(st.session_state, "cred_error") and st.session_state.cred_error != "":
        st.error(st.session_state.cred_error)

instructions_for_user = "Welcome to IBM Research Conversational Prompt Engineering (CPE) service.\n" \
            "This service is intended to help users build an effective prompt, tailored to their specific use case, through a simple chat with an LLM.\n" \
            "To make the most out of this service, it would be best to prepare in advance at least 3 input examples that represent your use case in a simple csv file. Alternatively, you can use sample data from our data catalog.\n" \
            "For more information feel free to contact us in slack via [#foundation-models-lm-utilization](https://ibm.enterprise.slack.com/archives/C04KBRUDR8R).\n"\
            "This assistant system uses BAM or CWatsonx to serve LLMs. Do not include PII or confidential information in your responses, nor in the data you share."

def init_set_up_page():
    st.title(":blue[IBM Research Conversational Prompt Engineering]")

    # default setting
    st.session_state.model = 'llama-3'
    if not hasattr(st.session_state, "target_model"):
        st.session_state.target_model = 'llama-3'

    load_environment_variables()

    credentials_are_set = 'credentials' in st.session_state and 'key' in st.session_state['credentials']
    email_is_set = hasattr(st.session_state, "email_address")
    OK_to_proceed_to_chat = credentials_are_set and email_is_set

    if OK_to_proceed_to_chat:
        return True

    else:
        st.empty()
        # with entry_page.form("my_form"):
        st.write(eval_instructions_for_user)

        only_watsonx = st.session_state["config"].getboolean("General", "only_watsonx")
        if not only_watsonx:
            api = st.radio(
                    "",
                    # add dummy option to make it the default selection
                options=["BAM", "Watsonx"] ,
                horizontal=True, key=f"bam_watsonx_radio",
                index=0 if st.session_state.API == APIName.BAM else 1)
        else:
            api = APIName.Watsonx
        st.session_state.API = APIName.BAM if api == "BAM" else APIName.Watsonx

        set_credentials()

        if USE_ONLY_LLAMA:
            target_model = 'llama-3'
        else:
            target_model = st.radio(
                label="Select the target model. The prompt that you will build will be formatted for this model.",
                options=["llama-3", "mixtral", "granite"],
                key="target_model_radio",
                captions=["llama-3-70B-instruct. ",
                          "mixtral-8x7B-instruct-v01. ",
                          "granite-13b-chat-v2  (Beta version)"])

        st.text_input(label="Organization email address", key="email_address_input")
        if hasattr(st.session_state, "email_error") and st.session_state.email_error != "":
            st.error(st.session_state.email_error)

        st.button("Submit", on_click=submit_button_clicked, args=[target_model])
        return False


def init_config():
    if len(sys.argv) > 1:
        logging.info(f"Loading {sys.argv[1]} config")
        config_name = sys.argv[1]
    else:
        logging.info(f"Loading default config")
        config_name = "main"

    config = load_config(config_name)
    st.session_state["config_name"] = config_name
    st.session_state["config"] = config
    st.session_state["dataset_name_to_dir"] = load_dataset_mapping(config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if not "config" in st.session_state:
        init_config()
    set_up_is_done = init_set_up_page()
    if set_up_is_done:
        callback_cycle()

