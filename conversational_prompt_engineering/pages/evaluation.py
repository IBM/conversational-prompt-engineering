import json
import os
from collections import Counter

import streamlit as st
import pandas as pd

from conversational_prompt_engineering.backend.prompt_building_util import build_few_shot_prompt
from conversational_prompt_engineering.backend.evaluation_core import Evaluation
from conversational_prompt_engineering.backend.llm_as_a_judge import LlmAsAJudge
from conversational_prompt_engineering.util.upload_csv_or_choose_dataset_component import create_choose_dataset_component_eval
import time

NUM_EXAMPLES = 5


dimensions = ["dim1", "dim2", "dim3"]

prompt_types = ["baseline", "zero_shot", "few_shot"]

work_modes = ["regular", "dummy_prompts"]
work_mode = 1

DEBUG_LLM_AS_A_JUDGE = False


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
    mixed_to_real = st.session_state.generated_data[st.session_state.count]["mixed_indices"][side]
    summary = st.session_state.generated_data[st.session_state.count][f"{mixed_to_real}_summary"]
    st.write(f"Summary {side+1}")
    st.text_area(label=f"output_{side}", value=summary, label_visibility="collapsed", height=200)


def display_llm_judge(side):
    mixed_to_real = st.session_state.generated_data[st.session_state.count]["mixed_indices"][side]
    judge = "ABS: " + ",".join(st.session_state.generated_data[st.session_state.count]['llm_judge'][f'{mixed_to_real}_llm_judge_abs']) \
            + "\n\n" + "REL BL_FS: " + \
            ",".join(st.session_state.generated_data[st.session_state.count]['llm_judge']['BL_FS_llm_judge_rel']) \
            + "\n\n" + "REL BL_ZS: " + \
            ",".join(st.session_state.generated_data[st.session_state.count]['llm_judge']['BL_ZS_llm_judge_rel']) \
            + "\n\n" + "REL ZS_FS: " + \
            ",".join(st.session_state.generated_data[st.session_state.count]['llm_judge']['ZS_FS_llm_judge_rel'])

    st.text_area(label=f"judge_{side}", value=judge, label_visibility="collapsed", height=200)


def calculate_results():
    ranked_elements = [d['prompts'] for d in st.session_state.generated_data if len(d['prompts']) > 0]
    prompts = {p : {} for p in prompt_types}
    for ranked_element in ranked_elements:
        for dimension, prompt_type in ranked_element.items():
            prompts[prompt_type][dimension] = prompts[prompt_type].get(dimension, 0) + 1

    return prompts, len(ranked_elements)


def save_results():
    out_path = os.path.join(st.session_state.manager.out_dir, "eval")
    os.makedirs(out_path, exist_ok=True)
    df = pd.DataFrame(st.session_state.generated_data)
    for dim in dimensions:
        for rank in ["Best", "Worst"]:
            df[f"ranked_prompt_{(dim,rank)}"] = df["prompts"].apply(lambda x: x.get((dim, rank)))
            df[f"sides_{(dim,rank)}"] = df["sides"].apply(lambda x: x.get((dim, rank)))
    df = df.drop(["sides", "prompts"], axis=1)
    df.to_csv(os.path.join(out_path, f"eval_results.csv"))
    with open(os.path.join(out_path, f"metadata.json"), "w") as f:
        prompts_dict = {}
        res_dict = {"dataset": st.session_state["selected_dataset"], "prompts" : prompts_dict}
        for i in range(len(st.session_state.eval_prompts)):
            prompts_dict[f"prompt_{i}"] = {"prompt_text": st.session_state.eval_prompts[i], "prompt_type": prompt_types[i]}
        json.dump(res_dict, f)

def process_user_selection():
    pass

def reset_evaluation():
    st.session_state.generated_data = []
    st.session_state.evaluate_clicked = False

def validate_annotation():
    for dim in dimensions:
        best = st.session_state.generated_data[st.session_state.count]["sides"][(dim, "Best")]
        worst = st.session_state.generated_data[st.session_state.count]["sides"][(dim, "Worst")]
        if (best == worst):
            st.error(f':heavy_exclamation_mark: You cannot select the same summary as best and worst in respect to {dim}')
            return False
    return True


def run():
    num_prompts = 0
    if 'manager' in st.session_state:
        num_prompts = len(st.session_state.manager.approved_prompts)

        if work_mode == 1 and num_prompts < 2:
            st.session_state.manager.prompts = ["output the line: we all live in a yellow submarine", "output the line: the long and winding road"]
            num_prompts = len(st.session_state.manager.approved_prompts)
            if st.session_state.manager.baseline_prompt == "":
                st.session_state.manager.baseline_prompt = "summarize the following text"

    if num_prompts < 1:
        st.write("Evaluation will be open after at least one prompt has been curated in the chat.")

    else:

        baseline_prompt = build_few_shot_prompt(st.session_state.manager.baseline_prompt, [],
                                                st.session_state.manager.bam_client.parameters['model_id'])
        zero_shot_prompt = build_few_shot_prompt(st.session_state.manager.approved_prompts[-1]['prompt'],
                                                 [],
                                                 st.session_state.manager.bam_client.parameters['model_id'])
        few_shot_examples = st.session_state.manager.approved_outputs[:st.session_state.manager.validated_example_idx]

        current_prompt = build_few_shot_prompt(st.session_state.manager.approved_prompts[-1 if work_mode == 0 else -2]['prompt'],
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
        if DEBUG_LLM_AS_A_JUDGE:
            test_texts = test_texts[:1]

        # get prompts to evaluate
        if 'evaluation' not in st.session_state:
            st.session_state.evaluation = Evaluation(st.session_state.manager.bam_client)

        if "llm_judge" not in st.session_state and DEBUG_LLM_AS_A_JUDGE:
            st.session_state.llm_judge = LlmAsAJudge(bam_api_key=st.session_state.key, model="prometheus_7b",
                                                     conv_id=st.session_state.conv_id, num_summaries=len(prompt_types))

        st.session_state.eval_prompts = [baseline_prompt, zero_shot_prompt, current_prompt]

        assert len(st.session_state.eval_prompts) == len(prompt_types), "number of prompts should be equal to the number of prompt types"
        if 'count' not in st.session_state:
            st.session_state.count = 0

        # show prompts
        prompt_cols = st.columns(len(prompt_types))
        prompt_text_area_titles = ["Prompt 1 (Baseline prompt)", "Prompt 2 (CPE zero shot prompt)", "Prompt 3 (CPE few shot prompt)"]
        assert (len(prompt_text_area_titles) == len(prompt_types))
        for i in range(len(prompt_types)):
            with prompt_cols[i]:
                st.write(prompt_text_area_titles[i])
                st.text_area(key=f"prompt_{i+1}", label="text", value=st.session_state.eval_prompts[i], label_visibility="collapsed", height=200)



        # show summarize button
        st.session_state.evaluate_clicked = False
        if test_texts is not None:
            st.session_state.evaluate_clicked = st.button("Generate outputs")

        # summarize texts using prompts
        if st.session_state.evaluate_clicked:
            with st.spinner('Generating outputs...'):
                generated_data = \
                    st.session_state.evaluation.summarize(st.session_state.eval_prompts, prompt_types,
                                                          test_texts)
                if "llm_judge" in st.session_state:
                    st.session_state.llm_judge.evaluate_prompt(zero_shot_prompt, generated_data)
                st.session_state.generated_data = generated_data
                for row in st.session_state.generated_data:
                    row['sides'] = {}
                    row['prompts'] = {}

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
            st.write("Bellow are presented the compared summaries. Please select the best and worst summary in respect to the different aspects. ")
            summary_cols_list = st.columns(len(prompt_types))

            for i in range(len(prompt_types)):
                with summary_cols_list[i]:
                    display_summary(i)
                    if "llm_judge" in st.session_state:
                        display_llm_judge(i)

            options = ["Best", "Worst"]
            radio_button_labels = [f"Summary {i+1}" for i in range(len(prompt_types))]
            for dim in dimensions:
                st.write(f"{dim}")
                cols = st.columns(len(options))
                for col, op in zip(cols, options):
                    with col:
                        selected_value = st.radio(
                                f"{op} summary:",
                            # add dummy option to make it the default selection
                                options = radio_button_labels,
                                horizontal=True, key=f"radio_{dim}_{op}",
                                index=st.session_state.generated_data[st.session_state.count]['sides'].get((dim,op)),
                                )
                        if selected_value:
                            side_index = radio_button_labels.index(selected_value)
                            mixed_to_real = st.session_state.generated_data[st.session_state.count]["mixed_indices"][side_index]
                            selected_prompt = st.session_state.generated_data[st.session_state.count][f"{mixed_to_real}_prompt"]
                            st.session_state.generated_data[st.session_state.count]['sides'][(dim,op)] = side_index
                            st.session_state.generated_data[st.session_state.count]['prompts'][(dim,op)] = mixed_to_real
                st.divider()

            num_of_answered_questions = len(st.session_state.generated_data[st.session_state.count]['prompts'])
            # enable only after the user ranked the all summaries
            finish_clicked = st.button(f"Submit annotation for example {st.session_state.count+1}", disabled = num_of_answered_questions != len(dimensions)*len(options))
            if finish_clicked:
                if validate_annotation():
                    # showing aggregated results
                    results, num_of_examples = calculate_results()
                    st.write(f"Finished annotating {num_of_examples} examples")
                    save_results()
                    st.write(f"Compared between {len(st.session_state.eval_prompts)} prompts")
                    for dim in dimensions:
                        st.write(f"{dim}:")
                        for prompt_type in results:
                            num_of_time_prompt_is_best = results[prompt_type].get((dim, "Best"), 0)
                            pct_val = '{0:.2f}'.format(100*num_of_time_prompt_is_best/num_of_examples)
                            st.write(f"{prompt_type} prompt was chosen {num_of_time_prompt_is_best} {'times' if num_of_time_prompt_is_best != 1 else 'time'} ({pct_val}%) ")


if __name__ == "__main__":
    run()
