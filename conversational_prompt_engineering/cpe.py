import os

import streamlit as st

from conversational_prompt_engineering.backend.double_chat_manager import DoubleChatManager

st.title("IBM Conversational Prompt Tuning")

assert 'BAM_APIKEY' in os.environ, 'please set the environment variable BAM_APIKEY'

if "manager" not in st.session_state:
    st.session_state.manager = DoubleChatManager()

out_messages = st.session_state.manager.process_user_input(st.chat_input(''))

for message in out_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

