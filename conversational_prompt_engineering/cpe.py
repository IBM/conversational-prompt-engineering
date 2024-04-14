import streamlit as st

from conversational_prompt_engineering.backend.manager import Manager, REQUEST_APIKEY_STRING

st.title("IBM Conversational Prompt Tuning")

manager = Manager()

if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': REQUEST_APIKEY_STRING}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# st.write("Hi, please provide your BAM API key")
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = manager.call(
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        )
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
