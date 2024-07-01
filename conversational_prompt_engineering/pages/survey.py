import streamlit as st
import os
import pandas as pd


questions = ["1.	I’m satisfied with the final prompt, it met my requirements. ",
             "2.	The system helped me think through how the summaries should look like and what criteria to consider when building the prompt",
             "3.	I felt the system was pleasant and responsive throughout the interaction.",
             "4.	I’m statisfied with the time it took to come up with the final prompt"
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
            index=None if answers[i] is None else answers[i]-1,
        )
        if selected_value:
            answers[i] = selected_value
    free_text = st.text_input(label = "Please write any other feedback here:", )
    submit_clicked = st.button("Submit")
    if submit_clicked:
        if None in answers:
            st.error(':heavy_exclamation_mark: Please respond to all the questions')
        else:
            save_survey(free_text)


if __name__ == "__main__":
    run()
