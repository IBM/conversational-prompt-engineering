import logging
import os
import time

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
    create_choose_dataset_component_train, add_evaluator_input
from st_pages import Page, show_pages, hide_pages

version = "callback manager v1.0.5"
st.set_page_config(layout="wide", menu_items={"About": f"CPE version: {version}"})

show_pages(
    [
        Page("cpe.py", "Chat", ""),
        Page("pages/evaluation.py", "Evaluate", ""),
        Page("pages/survey.py", "Survey", ""),

    ]
)


MUST_HAVE_UPLOADED_DATA_TO_START = True
USE_ONLY_LLAMA = True

class APIName(Enum):
    BAM, Watsonx = "bam", "watsonx"


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
        st.session_state.conv_id = sha1.hexdigest()[:16] # deterministic hash of 16 characters
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



def callback_cycle():
    # create the manager if necessary
    if "manager" not in st.session_state:
        sha1 = hashlib.sha1()
        sha1.update(st.session_state.credentials["key"].encode('utf-8'))
        st.session_state.conv_id = sha1.hexdigest()[:16]  # deterministic hash of 16 characters

        st.session_state.manager = CallbackChatManager(credentials=st.session_state.credentials, model=st.session_state.model,
                                                       target_model=st.session_state.target_model,
                                                       conv_id=st.session_state.conv_id, api = st.session_state.API.value)

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

        if os.path.exists(manager.result_json_file):
            with open(manager.result_json_file) as file:
                btn = st.download_button(
                    label="Download chat result",
                    data=file,
                    file_name=f'chat_result_{st.session_state["selected_dataset"]}.json',
                    mime="text/json"
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

def submit_button_clicked():
    creds_are_ok = True
    if st.session_state.API == APIName.BAM and st.session_state.bam_api_key != "":
        st.session_state.credentials = {'key': st.session_state.bam_api_key}
    elif st.session_state.API == APIName.Watsonx and st.session_state.watsonx_api_key != "" and st.session_state.project_id != "":
        st.session_state.credentials = {'key': st.session_state.watsonx_api_key, 'project_id': st.session_state.project_id}
    else:
        creds_are_ok = False
    if creds_are_ok:
        st.session_state.model = 'llama-3'
        st.session_state.target_model = model
    else:
        st.error(':heavy_exclamation_mark: Please provide your credentials')


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.title(":blue[IBM Research Conversational Prompt Engineering]")

eval_instructions = \
    "Welcome to IBM Research Conversational Prompt Engineering (CPE) service.\n\n" \
    "This service is intended to help users build an effective prompt, personalized to their specific summarization use case, through a simple chat with an LLM.\n\n" \
    f"The *prompts* built by CPE are comprised of two parts: an instruction, describing to the LLM in natural language how to generate the summaries; and up to 3 text-summary pairs, exemplifying how summaries should look like.\n\n" \
    "To use and evaluate CPE, proceed according to the following steps:\n\n" \
    "1.	After submitting your API key (see below), we will ask you to select a summarization dataset from our catalog. " \
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
    "To start, choose whether to interact over BAM or watsonx.\n\n" \
    "To obtain a BAM API key: navigate to [BAM](https://bam.res.ibm.com), login with your w3 id on the top-right side, and then copy the key from the box titled “Documentation”.\n\n" \
    "To obtain a watsonx API key and project id please follow these steps:\n\n" \
    "•	To create an API key, please see [documentation](https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui).\n\n" \
    "•	To find your project id, select the project from the [project list](https://dataplatform.cloud.ibm.com/projects/?context=wx), and then take the project id from Manage->General->Details.\n\n" \
    "Thank you for your time!"




if "BAM_APIKEY" in os.environ:
    st.session_state.credentials = {}
    st.session_state.API = APIName.BAM
    st.session_state.credentials["key"] = os.environ["BAM_APIKEY"]
elif "WATSONX_APIKEY" in os.environ:
    st.session_state.credentials = {}
    st.session_state.credentials = {"project_id": os.environ["PROJECT_ID"]}
    st.session_state.API = APIName.Watsonx
    st.session_state.credentials["key"] = os.environ["WATSONX_APIKEY"]
else:
    st.session_state.API = APIName.BAM  # default setting

#default setting
st.session_state.model = 'llama-3'
st.session_state.target_model = 'llama-3'

if 'credentials' not in st.session_state or 'key' not in st.session_state['credentials']:

        if 'credentials' not in st.session_state:
            st.session_state["credentials"] = {}
        entry_page = st.empty()
    #with entry_page.form("my_form"):
        st.write(eval_instructions)
        def set_credentials():
            st.session_state.API = APIName.Watsonx if api == "Watsonx" else APIName.BAM
            if st.session_state.API == APIName.Watsonx:
                key_val = st.session_state.credentials.get('key', None)
                st.text_input(label="Watsonx API key", key="watsonx_api_key", disabled=False, value=key_val)
                proj_id_val = st.session_state.credentials.get('project_id', None)
                st.text_input(label="project ID", key="project_id", disabled=False, value=proj_id_val)
            else:
                st.text_input(label="BAM API key", key="bam_api_key", disabled=False)

        api = st.radio(
            "",
            # add dummy option to make it the default selection
            options=["BAM", "Watsonx"],
            horizontal=True, key=f"bam_watsonx_radio",
            index=0 if st.session_state.API == APIName.BAM else 1)

        set_credentials()
        if USE_ONLY_LLAMA:
            model = 'llama-3'
        else:
            model = st.radio(label="Select the target model. The prompt that you will build will be formatted for this model.", options=["llama-3", "mixtral"],
                         captions=["llama-3-70B-instruct. Recommended for most use-cases.",
                                   "mixtral-8x7B-instruct-v01. Recommended for very long documents."])

        st.button("Submit", on_click=submit_button_clicked)

else:
    callback_cycle()
    # new_cycle()
    # old_cycle()


