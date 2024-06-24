import streamlit as st
import os


def save_survey(answers):
    out_path = os.path.join(st.session_state.manager.out_dir, "survey")
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "user_answers.txt"), "w") as f_out:
        for i, answer in enumerate(answers):
            f_out.write(f"{i+1}. {answer}\n")


def run():
    questions = ["question 1", "question 2"]
    answers = []
    for i, q in enumerate(questions):
        answers.append(st.text_input(label=q, key = f"survey_question_{i}"))

    submit_clicked = st.button("Submit")
    if submit_clicked:
        save_survey(answers)


if __name__ == "__main__":
    run()
