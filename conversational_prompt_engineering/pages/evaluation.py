import json
import os
from collections import Counter

import streamlit as st
import pandas as pd

from conversational_prompt_engineering.backend.prompt_building_util import build_few_shot_prompt, BASELINE_SUMMARIZATION_INSTRUCTION
from conversational_prompt_engineering.backend.evaluation_core import Evaluation
from conversational_prompt_engineering.util.upload_csv_or_choose_dataset_component import create_choose_dataset_component_eval
import time

NUM_EXAMPLES = 5


def display_text():
    text = st.session_state.generated_data[st.session_state.count]['text']
    st.text_area(label="text", value=text, label_visibility="collapsed", height=400)


def next_text():
    if st.session_state.count + 1 >= len(st.session_state.generated_data):
        st.session_state.count = 0
    else:
        st.session_state.count += 1


def previous_text():
    if st.session_state.count - 1 < 0:
        st.session_state.count = len(st.session_state.generated_data) - 1
    else:
        st.session_state.count -= 1


def display_summary(side):
    summary = st.session_state.generated_data[st.session_state.count][side]
    st.text_area(label=f"output_{side}", value=summary, label_visibility="collapsed", height=200)


def display_selected(selection):
    if st.session_state.generated_data[st.session_state.count]['selected_side'] and \
            st.session_state.generated_data[st.session_state.count]['selected_side'] == selection:
        st.write(":+1:")


def select(prompt, side):
    selected_prompt = "0" if prompt == st.session_state.prompts[0] else "1"
    st.session_state.generated_data[st.session_state.count]['selected_prompt'] = selected_prompt
    st.session_state.generated_data[st.session_state.count]['selected_side'] = side


def calculate_results():
    counter = Counter([d['selected_prompt'] for d in st.session_state.generated_data if d['selected_prompt'] is not None])
    return counter


def save_results():
    out_path = os.path.join(st.session_state.manager.out_dir, "eval")
    os.makedirs(out_path, exist_ok=True)
    df = pd.DataFrame(st.session_state.generated_data)
    df.to_csv(os.path.join(out_path, f"eval_results.csv"))
    with open(os.path.join(out_path, f"prompts.json"), "w") as f:
        json.dump({str(i): st.session_state.prompts[i] for i in range(len(st.session_state.prompts))}, f)


def reset_evaluation():
    st.session_state.generated_data = []
    st.session_state.evaluate_clicked = False


def run():
    num_prompts = 0
    if 'manager' in st.session_state:
        num_prompts = len(st.session_state.manager.approved_prompts)
    if num_prompts < 1:
        st.write("Evaluation will be open after at least one prompt has been curated in the chat.")
    else:

        baseline_prompt = build_few_shot_prompt(BASELINE_SUMMARIZATION_INSTRUCTION, [],
                                                st.session_state.manager.bam_client.parameters['model_id'])
        zero_shot_prompt = build_few_shot_prompt(st.session_state.manager.approved_prompts[-1]['prompt'],
                                                 [],
                                                 st.session_state.manager.bam_client.parameters['model_id'])
        few_shot_examples = st.session_state.manager.approved_outputs[:st.session_state.manager.validated_example_idx]
        current_prompt = build_few_shot_prompt(st.session_state.manager.approved_prompts[-1]['prompt'],
                                               few_shot_examples,
                                               st.session_state.manager.bam_client.parameters['model_id'])

        # present instructions
        st.title("IBM Research Conversational Prompt Engineering - Evaluation")
        with st.expander("Instructions (click to expand)"):
            st.markdown(f"1) In case you built the prompt using your own data rather than a datasets from our catalog, you should upload a csv file with the evaluation examples. {NUM_EXAMPLES} examples are chosen at random for evaluation.")
            st.markdown("2) Below you can see the prompts that were curated during your chat and will be used for evaluation.")
            st.markdown(f"3) Next, click on ***Generate output***. Each prompt will be used to generate an output for each of the examples.")
            st.markdown("4) After the outputs are generated, select the best output for each text. The order of the outputs is mixed for each example.")
            st.markdown("5) When you are done, click on ***Submit*** to present the evaluation scores.")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Reset evaluation"):
                reset_evaluation()

            st.write(f"Using model [{st.session_state.manager.bam_client.parameters['model_id']}](https://bam.res.ibm.com/docs/models#{st.session_state.manager.bam_client.parameters['model_id'].replace('/', '-')})")

        test_texts = create_choose_dataset_component_eval(st)

        col1, col2 = st.columns(2)
        with col1:
            # option = st.selectbox(
            #     "Select prompt to compare",
            #     options=["Baseline prompt", "Zero-shot prompt"])
            option = "Baseline prompt"
            if option == "Baseline prompt":
                prompt_to_compare = baseline_prompt
            else: # option == "Zero-shot prompt":
                prompt_to_compare = zero_shot_prompt
            if prompt_to_compare == current_prompt:
                st.write("Note: prompt 1 is identical to prompt 2")

        # get prompts to evaluate
        if 'evaluation' not in st.session_state:
            st.session_state.evaluation = Evaluation(st.session_state.manager.bam_client)

        st.session_state.prompts = [prompt_to_compare, current_prompt]

        if 'count' not in st.session_state:
            st.session_state.count = 0

        # show prompts
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Prompt 1 ({option})")
            st.text_area(key="prompt_1", label="text", value=st.session_state.prompts[0], label_visibility="collapsed", height=200)

        with col2:
            st.write("Prompt 2 (latest cpe prompt)")
            st.text_area(label="text", value=st.session_state.prompts[1], label_visibility="collapsed", height=200)

        # show summarize button
        st.session_state.evaluate_clicked = False
        if test_texts is not None:
            st.session_state.evaluate_clicked = st.button("Generate outputs")

        # summarize texts using prompts
        if st.session_state.evaluate_clicked:
            with st.spinner('Generating outputs...'):
                generated_data_mixed, generated_data_ordered = \
                    st.session_state.evaluation.summarize(st.session_state.prompts,
                                                          test_texts)
                st.session_state.generated_data = generated_data_mixed
                for row in st.session_state.generated_data:
                    row['selected_side'] = None
                    row['selected_prompt'] = None

        # showing texts and summaries to evaluate
        if 'generated_data' in st.session_state and len(st.session_state.generated_data) > 0:
            st.header(f"Text {st.session_state.count+1}/{len(st.session_state.generated_data)}", divider="gray")
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            with col1:
                if st.button("⏮️ Previous", on_click=previous_text):
                    pass
            with col2:
                if st.button("Next ⏭️", on_click=next_text):
                    pass
            display_text()
            st.divider()
            st.subheader("Generated outputs (random order)")
            col1, col2 = st.columns(2)
            with col1:
                display_summary("0")
                if st.button("Select", key="left", on_click=select, args=(st.session_state.generated_data[st.session_state.count]["0_prompt"], "0", )):
                    pass
                display_selected("0")

            with col2:
                display_summary("1")
                if st.button("Select", key="right", on_click=select, args=(st.session_state.generated_data[st.session_state.count]["1_prompt"], "1", )):
                    pass
                display_selected("1")
            # if all([row['selected_prompt'] for row in st.session_state.generated_data]):
            st.divider()
            finish_clicked = st.button("Submit")
            if finish_clicked:
                # showing aggregated results
                results = calculate_results()
                save_results()
                st.write(f"Compared between {option} and the latest cpe prompt")
                total_votes = sum(results.values())
                for item in results.most_common():
                    pct_val = '{0:.2f}'.format(100*item[1]/total_votes)
                    st.write(f"Prompt {int(item[0])+1} was chosen {item[1]} {'times' if item[1] > 1 else 'time'} ({pct_val}%)")


if __name__ == "__main__":
    run()
