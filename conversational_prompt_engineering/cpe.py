import logging
import os
import time

import pandas as pd
import streamlit as st
import hashlib
from genai.schema import ChatRole
from streamlit_js_eval import streamlit_js_eval

from conversational_prompt_engineering.backend.callback_chat_manager import CallbackChatManager
from conversational_prompt_engineering.backend.double_chat_manager import DoubleChatManager
from conversational_prompt_engineering.backend.manager import Manager, Mode
from conversational_prompt_engineering.util.upload_csv_or_choose_dataset_component import \
    create_choose_dataset_component_train
from st_pages import Page, show_pages, hide_pages

version = "callback manager v1.0.1"
st.set_page_config(layout="wide", menu_items={"About": f"CPE version: {version}"})

show_pages(
    [
        Page("cpe.py", "Chat", ""),
        Page("pages/evaluation.py", "Evaluate", ""),
    ]
)


def old_reset_chat():
    st.session_state.manager = Manager(st.session_state.mode, st.session_state.key)
    st.session_state.messages = []


def reset_chat():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


def new_cycle():
    # 1. create the manager if necessary
    if "manager" not in st.session_state:
        st.session_state.conv_id = abs(hash(st.session_state.key))
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
        sha1.update(st.session_state.key.encode('utf-8'))
        st.session_state.conv_id = sha1.hexdigest()[:16]  # deterministic hash of 16 characters

        st.session_state.manager = CallbackChatManager(bam_api_key=st.session_state.key, model=st.session_state.model,
                                                       conv_id=st.session_state.conv_id)
        st.session_state.manager.add_welcome_message()

    manager = st.session_state.manager

    # layout reset and upload buttons in 3 columns
    if st.button("Reset chat"):
        streamlit_js_eval(js_expressions="parent.window.location.reload()")

    uploaded_file = create_choose_dataset_component_train(st=st, manager=manager)
    if uploaded_file:
        manager.add_user_message("Selected data")

    if user_msg := st.chat_input("Write your message here"):
        manager.add_user_message(user_msg)

    for msg in manager.user_chat[:manager.user_chat_length]:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # generate and render the agent response
    messages = manager.generate_agent_messages()
    for msg in messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])


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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.title(":blue[IBM Research Conversational Prompt Engineering]")
if 'BAM_APIKEY' in os.environ:
    st.session_state['key'] = os.environ['BAM_APIKEY']
    st.session_state.model = 'llama3'

if 'BAM_APIKEY' not in os.environ and "key" not in st.session_state:
    entry_page = st.empty()
    with entry_page.form("my_form"):
        st.write("Welcome to IBM Research Conversational Prompt Engineering (CPE) service.")
        st.write(
            "This service is intended to help users build an effective prompt, tailored to their specific summarization use case, through a simple chat with an LLM.")
        st.write(
            "To make the most out of this service, it would be best to prepare in advance at least 3 input examples that represent your use case in a simple csv file.")
        st.write(
            "For more information feel free to contact us in slack via [#foundation-models-lm-utilization](https://ibm.enterprise.slack.com/archives/C04KBRUDR8R).")
        st.write(
            "This assistant system uses BAM to serve LLMs. Do not include PII or confidential information in your responses, nor in the data you share.")
        st.write("To proceed, please provide your BAM API key and select a model.")
        key = st.text_input(label="BAM API key")
        model = st.radio(label="Select model", options=["llama3", "mixtral"],
                         captions=["Recommended for most use-cases",
                                   "Recommended for very long documents"])
        submit = st.form_submit_button()
        if submit:
            if len(key) != 0:
                st.session_state.key = key
                st.session_state.model = model
                entry_page.empty()
            else:
                st.error(':heavy_exclamation_mark: You cannot proceed without providing your BAM API key')

if 'key' in st.session_state:
    callback_cycle()
    # new_cycle()
    # old_cycle()

