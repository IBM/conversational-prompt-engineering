import logging
import os

import streamlit as st

from conversational_prompt_engineering.backend.double_chat_manager import DoubleChatManager
from conversational_prompt_engineering.backend.manager import Manager, Mode


def reset_chat():
    st.session_state.manager = Manager(st.session_state.mode, st.session_state.key)
    st.session_state.messages = []


def new_cycle():
    if "manager" not in st.session_state:
        st.session_state.manager = DoubleChatManager(bam_api_key=st.session_state.key)

    out_messages = st.session_state.manager.process_user_input(st.chat_input(''))

    for message in out_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


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
            reset_chat()

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
        st.write("Do not share any confidential information in this conversation")
        st.write("To proceed, please provide your BAM API key")
        key = st.text_input(label="BAM API key")
        submit = st.form_submit_button()
        if submit:
            st.session_state.key = key

if "key" in st.session_state:
    if st.button("Reset chat"):
        reset_chat()

# new_cycle()
old_cycle()
