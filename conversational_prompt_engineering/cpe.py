import os

import streamlit as st
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from conversational_prompt_engineering.backend.manager import Manager, REQUEST_APIKEY_STRING, Mode

st.title("IBM Conversational Prompt Engineering")


def reset_chat():
    st.session_state.manager = Manager(st.session_state.mode)
    if 'BAM_APIKEY' not in os.environ:
        st.session_state.messages = [{'role': 'assistant', 'content': REQUEST_APIKEY_STRING}]
    else:
        st.session_state.messages = []


if st.button("Reset chat"):
    reset_chat()

mode = st.radio(label="Mode", options=["Basic", "Advanced"],
                captions=["basic zero-shot -> few-shot (default)", "basic zero-shot -> custom zero-shot -> few-shot"])

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
    st.session_state.manager = Manager(st.session_state.mode)

if "messages" not in st.session_state:
    if 'BAM_APIKEY' not in os.environ:
        st.session_state.messages = [{'role': 'assistant', 'content': REQUEST_APIKEY_STRING}]
    else:
        st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# st.write("Hi, please provide your BAM API key")
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
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
    st.session_state.messages.append({"role": "assistant", "content": response})
