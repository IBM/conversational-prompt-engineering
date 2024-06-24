import json
import os
from collections import Counter

import streamlit as st
import pandas as pd

from conversational_prompt_engineering.backend.prompt_building_util import build_few_shot_prompt
from conversational_prompt_engineering.backend.evaluation_core import Evaluation
from conversational_prompt_engineering.util.upload_csv_or_choose_dataset_component import create_choose_dataset_component_eval
import time

NUM_EXAMPLES = 5

NUM_PROMPTS_TO_COMPARE = 3

RANKING_BUTTON_TITLES = ["Select best", "Select second best", "Third place"]

assert (len(RANKING_BUTTON_TITLES), NUM_PROMPTS_TO_COMPARE)

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
    pass
    #if st.session_state.generated_data[st.session_state.count]['selected_side'] and \
    #        st.session_state.generated_data[st.session_state.count]['selected_side'] == selection:
    #    st.write(":+1:")


def select(prompt, side_idx):
    def set_rank(rank, selected_prompt_index, side_idx):
        st.session_state.generated_data[st.session_state.count]['ranked_prompts'][selected_prompt_index] = rank
        st.session_state.generated_data[st.session_state.count]['ranked_sides'][side_idx] = rank

    #prompt index is not the same as the side index because the prompts are shuffled
    selected_prompt = st.session_state.prompts.index(prompt)
    # we select either the best (rank 0) or the second best (rank 1)
    curr_rank = 0 if len(st.session_state.generated_data[st.session_state.count]['ranked_prompts']) == 0 else 1
    set_rank(curr_rank, selected_prompt, side_idx)
    if curr_rank == 1:
        # set the last prompt
        missing_prompt_idx = [x for x in range(NUM_PROMPTS_TO_COMPARE) if x not in
                            st.session_state.generated_data[st.session_state.count]['ranked_prompts']]
        missing_side_idx = [x for x in range(NUM_PROMPTS_TO_COMPARE) if x not in
                            st.session_state.generated_data[st.session_state.count]['ranked_sides']]
        assert (len(missing_side_idx) == len(missing_side_idx) == 1)
        set_rank(2, missing_prompt_idx[0], missing_side_idx[0])




def clear_ranking():
    st.session_state.generated_data[st.session_state.count]['ranked_prompts'].clear()
    st.session_state.generated_data[st.session_state.count]['ranked_sides'].clear()


def calculate_results():
    ranked_elements = [d['ranked_prompts'] for d in st.session_state.generated_data if len(d['ranked_prompts']) > 0]
    # generate a NUM_PROMPTS_TO_COMPARE x NUM_PROMPTS_TO_COMPARE map
    prompts = {x : {y: 0 for y in range(NUM_PROMPTS_TO_COMPARE)} for x in range(NUM_PROMPTS_TO_COMPARE)}
    for ranked_element in ranked_elements:
        for prompt_idx in range(len(st.session_state.prompts)):
            prompts[prompt_idx][ranked_element[prompt_idx]] += 1

    return prompts, len(ranked_element)


def save_results():
    out_path = os.path.join(st.session_state.manager.out_dir, "eval")
    os.makedirs(out_path, exist_ok=True)
    df = pd.DataFrame(st.session_state.generated_data)
    for i in range(NUM_PROMPTS_TO_COMPARE):
        df[f"ranked_prompt_{i}"] = df["ranked_prompts"].apply(lambda x: x.get(i, 0))
        df[f"ranked_sides_{i}"] = df["ranked_sides"].apply(lambda x: x.get(i, 0))
    df = df.drop(["ranked_sides", "ranked_prompts"], axis=1)
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

        #TODO lena: remove this before commit!!!!!
        if num_prompts < 1:
            st.session_state.manager.prompts = ["output the line: we all live in a yellow submarine", "output the line: the long and winding road"]
            num_prompts = len(st.session_state.manager.approved_prompts)

    if num_prompts < 1:
        st.write("Evaluation will be open after at least one prompt has been curated in the chat.")

    else:

        baseline_prompt = build_few_shot_prompt(st.session_state.manager.baseline_prompt, [],
                                                st.session_state.manager.bam_client.parameters['model_id'])
        zero_shot_prompt = build_few_shot_prompt(st.session_state.manager.approved_prompts[-1]['prompt'],
                                                 [],
                                                 st.session_state.manager.bam_client.parameters['model_id'])
        few_shot_examples = st.session_state.manager.approved_outputs[:st.session_state.manager.validated_example_idx]
        #TODO lena: move back -2 to -1
        current_prompt = build_few_shot_prompt(st.session_state.manager.approved_prompts[-2]['prompt'],
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

        # get prompts to evaluate
        if 'evaluation' not in st.session_state:
            st.session_state.evaluation = Evaluation(st.session_state.manager.bam_client)

        st.session_state.prompts = [baseline_prompt, zero_shot_prompt, current_prompt]

        if 'count' not in st.session_state:
            st.session_state.count = 0

        # show prompts
        prompt_cols = st.columns(NUM_PROMPTS_TO_COMPARE)
        prompt_text_area_titles = ["Prompt 1 (Baseline prompt)", "Prompt 2 (CPE zero shot prompt)", "Prompt 3 (CPE few shot prompt)"]
        assert (len(prompt_text_area_titles) == NUM_PROMPTS_TO_COMPARE)
        for i in range(NUM_PROMPTS_TO_COMPARE):
            with prompt_cols[i]:
                st.write(prompt_text_area_titles[i])
                st.text_area(key=f"prompt_{i+1}", label="text", value=st.session_state.prompts[i], label_visibility="collapsed", height=200)



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
                    row['ranked_sides'] = {}
                    row['ranked_prompts'] = {}

        # showing texts and summaries to evaluate
        if 'generated_data' in st.session_state and len(st.session_state.generated_data) > 0:
            st.header(f"Text {st.session_state.count+1}/{len(st.session_state.generated_data)}", divider="gray")
            col1, col2, col3, col4, col5 = st.columns([1]*5)
            with col1:
                if st.button("⏮️ Previous", on_click=previous_text):
                    pass
            with col2:
                if st.button("Next ⏭️", on_click=next_text):
                    pass
            display_text()
            st.divider()
            st.subheader("Generated outputs (random order)")
            summary_cols_list = st.columns(NUM_PROMPTS_TO_COMPARE)
            ranked_sides = st.session_state.generated_data[st.session_state.count]['ranked_sides']
            button_titles = {x : RANKING_BUTTON_TITLES[0] for x in range(NUM_PROMPTS_TO_COMPARE)}
            button_disabled = {x: False for x in range(NUM_PROMPTS_TO_COMPARE)}

            # set up ranking buttons
            num_of_already_ranked = len(ranked_sides)
            for x in range(NUM_PROMPTS_TO_COMPARE):
                if x in ranked_sides:
                    button_disabled[x] = True
                    button_titles[x] = RANKING_BUTTON_TITLES[ranked_sides[x]] #if it's ranked 0, the button should have the title RANKING_BUTTON_TITLES[0]
                else:
                    button_titles[x] = RANKING_BUTTON_TITLES[num_of_already_ranked] #the rest of the buttons should have the first title that wasn't used yet

            for i in range(NUM_PROMPTS_TO_COMPARE):
                with summary_cols_list[i]:
                    display_summary(f"{i}")
                    if st.button(button_titles[i], key=f"{i}_summary", on_click=select, disabled = button_disabled[i],
                                 args=(st.session_state.generated_data[st.session_state.count][f"{i}_prompt"], i, )):
                        pass
                    display_selected(f"{i}")


            # if all([row['selected_prompt'] for row in st.session_state.generated_data]):
            st.divider()
            ranked_elements_num = len(st.session_state.generated_data[st.session_state.count]['ranked_sides'])

            st.button("Clear selection", on_click=clear_ranking, disabled= ranked_elements_num == 0)
            # enable only after the user ranked the all summaries
            finish_clicked = st.button("Submit", disabled=ranked_elements_num != NUM_PROMPTS_TO_COMPARE)
            if finish_clicked:
                # showing aggregated results
                results, num_of_examples = calculate_results()
                save_results()
                st.write(f"Compared between {len(st.session_state.prompts)} prompts")
                for prompt_id in results:
                    num_of_time_prompt_is_best = results[prompt_id][0]
                    pct_val = '{0:.2f}'.format(100*num_of_time_prompt_is_best/num_of_examples)
                    st.write(f"Prompt {int(prompt_id)+1} was chosen {num_of_time_prompt_is_best} {'times' if num_of_time_prompt_is_best != 1 else 'time'} ({pct_val}%)")


if __name__ == "__main__":
    run()
