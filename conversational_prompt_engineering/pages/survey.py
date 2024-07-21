import streamlit as st
import os
import pandas as pd


def get_chosen_prompt():
    if hasattr(st.session_state, "manager") and hasattr(st.session_state.manager, "prompts"):
        return st.session_state.manager.prompts[-1] if st.session_state.manager.prompts \
    else "no-prompt"

prompt_from_chat = get_chosen_prompt()
questions = [f"1.	I’m satisfied with the final prompt **{prompt_from_chat}**, it met my requirements. ",
             "2.	The system helped me think through how the desired outputs should look like and what criteria to consider when building the prompt.",
             "3.	I felt the system was pleasant and responsive throughout the interaction.",
             "4.	I’m satisfied with the time it took to come up with the final prompt."
             ]
answers = [None]* len(questions)

def save_survey(free_text):
    out_path = os.path.join(st.session_state.manager.out_dir, "survey")
    os.makedirs(out_path, exist_ok=True)
    df = pd.DataFrame({f"q_{i}" : [answers[i] ]for i in range(len(questions))})
    df[f"q_{len(answers)}"] = free_text
    df.to_csv(os.path.join(out_path, "survey.csv"))


def run():
    radio_button_options = [1, 2, 3, 4, 5]
    st.write("Please rate your agreement with the following statements (1 – Strongly disagree, 5 – Strongly agree)")
    for i, q in enumerate(questions):
        selected_value = st.radio(
            f"{q}",
            # add dummy option to make it the default selection
            options=radio_button_options,
            horizontal=True, key=f"summary_radio_{i}",
            index=None
        )
        if selected_value:
            answers[i] = selected_value
    free_text = st.text_area(label = "Please write any other feedback here:", height=150)
    submit_clicked = st.button("Submit")
    if submit_clicked:
        if None in answers:
            st.error(':heavy_exclamation_mark: Please respond to all the questions')
        else:
            save_survey(free_text)
            st.session_state["survey_is_submitted"] = True
            st.write("Thanks for responding!")


if __name__ == "__main__":
    if not hasattr(st.session_state, "manager") or not hasattr(st.session_state.manager, "prompt_conv_end") or not st.session_state.manager.prompt_conv_end:
        st.write("Survey will be open after at least one prompt has been curated in the chat.")
    else:
        if hasattr(st.session_state, "survey_is_submitted"):
            st.write("Survey was already submitted.")
        else:
            run()
