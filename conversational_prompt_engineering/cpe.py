import logging
import os

import pandas as pd
import streamlit as st

from conversational_prompt_engineering.backend.double_chat_manager import DoubleChatManager
from conversational_prompt_engineering.backend.manager import Manager, Mode


def old_reset_chat():
    st.session_state.manager = Manager(st.session_state.mode, st.session_state.key)
    st.session_state.messages = []


def reset_chat():
    st.session_state.manager = DoubleChatManager(bam_api_key=st.session_state.key)
    st.session_state.messages = []


def new_cycle():
    # 1. create the manager if necessary
    if "manager" not in st.session_state:
        st.session_state.manager = DoubleChatManager(bam_api_key=st.session_state.key)
    manager = st.session_state.manager

    # 2. layout reset and upload buttons in 2 columns
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reset chat"):
            reset_chat()
    with col2:
        if uploaded_file := st.file_uploader("Upload text examples csv"):
            manager.process_examples(pd.read_csv(uploaded_file))

    # 3. user input
    if user_msg := st.chat_input("Write your message here"):
        manager.add_user_message(user_msg)

    # 4. render the existing messages
    for msg in manager.user_chat:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    # 5. generate and render the agent response
    manager.generate_agent_message()
    msg = manager.user_chat[-1]
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

st.title("IBM Research Conversational Prompt Engineering")
if 'BAM_APIKEY' in os.environ:
    st.session_state['key'] = os.environ['BAM_APIKEY']

if 'BAM_APIKEY' not in os.environ and "key" not in st.session_state:
    with st.form("my_form", clear_on_submit=True):
        st.write("Welcome to IBM Research LMU CPE")
        st.write("This assistant system uses BAM to serve LLMs. Do not include PII or confidential information in your responses.")
        st.write("To proceed, please provide your BAM API key")
        key = st.text_input(label="BAM API key")
        submit = st.form_submit_button()
        if submit:
            st.session_state.key = key

if 'key' in st.session_state:
    new_cycle()
    # old_cycle()
