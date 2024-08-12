import streamlit as st
import streamlit as st




q_n_a = [("What is the purpose of CPE?",
                    "CPE is designed to create prompts for recurring tasks. In a conversational process, CPE assists in generating a prompt tailored to your task. You will see output examples based on the text you provide, allowing you to refine your prompt accordingly. Ultimately, you'll have a prompt that can be applied to various text examples.") ,
         ("What is the format of the expected input?",
                    "You are required to provide at least 3 input examples for the prompt-building phase and at least 5 examples for the evaluation. The expected format is a CSV file with one example per row, with a column titled \"text\" containing the input texts."),
         ("How can I see the full example text during the discussion of the generated output for this example?",
                    "In the message box, locate the question mark. When you hover over it, a tooltip will appear displaying the full text.") ]

def run():
    for i, q in enumerate(q_n_a):
        with st.chat_message("user"):
            st.markdown(q[0])
        with st.chat_message("assistant"):
            st.markdown(q[1])



if __name__ == "__main__":
    run()
