import json
import os
from collections import Counter

import streamlit as st
import pandas as pd

from enum import Enum
from conversational_prompt_engineering.backend.prompt_building_util import build_few_shot_prompt
from conversational_prompt_engineering.backend.evaluation_core import Evaluation
from conversational_prompt_engineering.backend.llm_as_a_judge import LlmAsAJudge
from conversational_prompt_engineering.util.upload_csv_or_choose_dataset_component import create_choose_dataset_component_eval
import time

MIN_NUM_EXAMPLES_TO_UPLOAD = 5
MIN_EXAMPLE_TO_EVALUATE = 3

class WorkMode(Enum):
    REGULAR, DUMMY_PROMPT = range(2)

work_mode = WorkMode.REGULAR

dimensions = ["dim1"]

prompt_types = ["baseline", "zero_shot", "few_shot"]

def build_baseline_prompt():
    return build_few_shot_prompt(st.session_state.manager.baseline_prompts["user_baseline_prompt"], [],
                          st.session_state.manager.target_bam_client.parameters['model_id'])

def build_z_sh_prompt():
    return build_few_shot_prompt(st.session_state.manager.approved_prompts[-1]['prompt'],
                                                 [],
                                                 st.session_state.manager.target_bam_client.parameters['model_id'])

def build_f_sh_prompt():
    few_shot_examples = st.session_state.manager.approved_outputs[:st.session_state.manager.validated_example_idx]

    return build_few_shot_prompt(
        st.session_state.manager.approved_prompts[-2 if work_mode == WorkMode.DUMMY_PROMPT else -1]['prompt'],
        few_shot_examples,
        st.session_state.manager.target_bam_client.parameters['model_id'])

prompt_type_metadata = {"baseline": {"title": "Prompt 1 (Baseline prompt)", "build_func": build_baseline_prompt},
                        "zero_shot": {"title": "Prompt 2 (CPE zero shot prompt)", "build_func": build_z_sh_prompt},
                         "few_shot": {"title": "Prompt 3 (CPE few shot prompt)", "build_func": build_f_sh_prompt}}



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


def display_output(side):
    mixed_to_real = st.session_state.generated_data[st.session_state.count]["mixed_indices_mapping_to_prompt_type"][side]
    output = st.session_state.generated_data[st.session_state.count][f"{mixed_to_real}_output"]
    st.write(f"Output {side+1}")
    st.text_area(label=f"output_{side}", value=output, label_visibility="collapsed", height=200)


def display_llm_judge(side):
    mixed_to_real = st.session_state.generated_data[st.session_state.count]["mixed_indices_mapping_to_prompt_type"][side]
    judge = "ABS: " + \
            st.session_state.generated_data[st.session_state.count][f'{mixed_to_real}_llm_judge_abs_result'] \
            + " Feedback: " + \
            st.session_state.generated_data[st.session_state.count][f'{mixed_to_real}_llm_judge_abs_feedback'] \
            + "\n\n" + "REL BL_FS: " + \
            st.session_state.generated_data[st.session_state.count]['BL_FS_llm_judge_rel_result'] \
            + " Feedback: " + \
            st.session_state.generated_data[st.session_state.count]['BL_FS_llm_judge_rel_feedback'] \
            + "\n\n" + "REL BL_ZS: " + \
            st.session_state.generated_data[st.session_state.count]['BL_ZS_llm_judge_rel_result'] \
            + " Feedback: " + \
            st.session_state.generated_data[st.session_state.count]['BL_ZS_llm_judge_rel_feedback'] \
            + "\n\n" + "REL ZS_FS: " + \
            st.session_state.generated_data[st.session_state.count]['ZS_FS_llm_judge_rel_result'] \
            + " Feedback: " + \
            st.session_state.generated_data[st.session_state.count]['ZS_FS_llm_judge_rel_feedback']

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
    #reorder shuffled data:
    ordered_generate_data = sorted(st.session_state.generated_data, key=lambda x: x["index"])

    df = pd.DataFrame(ordered_generate_data)
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
            st.error(f':heavy_exclamation_mark: You cannot select the same output as best and worst in respect to {dim}')
            return False
    return True


def run():
    num_prompts = 0

    if 'manager' in st.session_state:
        num_prompts = len(st.session_state.manager.approved_prompts)

        if work_mode == WorkMode.DUMMY_PROMPT and num_prompts < 2:
            st.session_state.manager.prompts = ["output the line: we all live in a yellow submarine", "output the line: the long and winding road"]
            num_prompts = len(st.session_state.manager.approved_prompts)
            if "model_baseline_prompt" not in st.session_state.manager.baseline_prompts:
                st.session_state.manager.baseline_prompts["model_baseline_prompt"] = "summarize the following text"

    if num_prompts < 1:
        st.write("Evaluation will be open after at least one prompt has been curated in the chat.")

    else:
        st.session_state.eval_prompts = []
        for prompt_type in prompt_types:
            st.session_state.eval_prompts.append(prompt_type_metadata[prompt_type]["build_func"]())



        # present instructions
        st.title("IBM Research Conversational Prompt Engineering - Evaluation")
        with st.expander("Instructions (click to expand)"):
            st.markdown(f"1) In case you built the prompt using your own data rather than a datasets from our catalog, you should upload a csv file with the evaluation examples. {MIN_NUM_EXAMPLES_TO_UPLOAD} examples are chosen at random for evaluation.")
            st.markdown("2) Below you can see the prompts that were curated during your chat and will be used for evaluation.")
            st.markdown(f"3) Next, click on ***Generate output***. Each prompt will be used to generate an output for each of the examples.")
            st.markdown("4) After the outputs are generated, select the best output for each text. The order of the outputs is mixed for each example.")
            st.markdown("5) When you are done, click on ***Submit*** to present the evaluation scores.")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Reset evaluation"):
                reset_evaluation()

            st.write(f"Using model [{st.session_state.manager.target_bam_client.parameters['model_id']}](https://bam.res.ibm.com/docs/models#{st.session_state.manager.target_bam_client.parameters['model_id'].replace('/', '-')})")

        test_texts = create_choose_dataset_component_eval(st)
        if DEBUG_LLM_AS_A_JUDGE:
            test_texts = test_texts[:1]

        # get prompts to evaluate
        if 'evaluation' not in st.session_state:
            st.session_state.evaluation = Evaluation(st.session_state.manager.target_bam_client)

        if "llm_judge" not in st.session_state and DEBUG_LLM_AS_A_JUDGE:
            st.session_state.llm_judge = LlmAsAJudge(bam_api_key=st.session_state.key, model="prometheus_7b",
                                                     conv_id=st.session_state.conv_id)

        assert len(st.session_state.eval_prompts) == len(prompt_types), "number of prompts should be equal to the number of prompt types"
        if 'count' not in st.session_state:
            st.session_state.count = 0

        # show prompts
        prompt_cols = st.columns(len(prompt_types))
        for i in range(len(prompt_types)):
            with prompt_cols[i]:
                st.write(prompt_type_metadata.get(prompt_types[i])["title"])
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
                    assert "zero_shot" in prompt_types, "cannot run llm as a judge without a zero shot prompt!"
                    zero_shot_prompt = st.session_state.eval_prompts[prompt_types.index("zero_shot")]
                    st.session_state.llm_judge.evaluate_prompt(zero_shot_prompt, generated_data)
                st.session_state.generated_data = generated_data
                for row in st.session_state.generated_data:
                    row['sides'] = {}
                    row['prompts'] = {}

        def add_next_buttons(s):
            col1, col2, col3, col4, col5 = st.columns([1]*5)
            with col1:
                if st.button("⏮️ Previous", on_click=previous_text, key=f"prev_{s}"):
                    pass
            with col2:
                if st.button("Next ⏭️", on_click=next_text, key=f"next_{s}"):
                    pass


        # showing texts and summaries to evaluate
        if 'generated_data' in st.session_state and len(st.session_state.generated_data) > 0:
            st.header(f"Text {st.session_state.count+1}/{len(st.session_state.generated_data)}", divider="gray")
            add_next_buttons("above_summaries")
            display_text()
            st.divider()
            st.subheader("Generated outputs (random order)")
            st.write("Bellow are presented the compared summaries. Please select the best and worst output in respect to the different aspects. ")
            output_cols_list = st.columns(len(prompt_types))

            for i in range(len(prompt_types)):
                with output_cols_list[i]:
                    display_output(i)
                    if "llm_judge" in st.session_state:
                        display_llm_judge(i)
            add_next_buttons("bellow_summaries")
            options = ["Best", "Worst"]
            radio_button_labels = [f"Output {i+1}" for i in range(len(prompt_types))]
            for dim in dimensions:
                st.write(f"{dim}")
                cols = st.columns(len(options))
                for col, op in zip(cols, options):
                    with col:
                        selected_value = st.radio(
                                f"{op} output:",
                            # add dummy option to make it the default selection
                                options = radio_button_labels,
                                horizontal=True, key=f"radio_{st.session_state.count}_{dim}_{op}",
                                index=st.session_state.generated_data[st.session_state.count]['sides'].get((dim,op))
                                )
                        if selected_value:
                            side_index = radio_button_labels.index(selected_value)
                            real_prompt_type = st.session_state.generated_data[st.session_state.count]["mixed_indices_mapping_to_prompt_type"][side_index]
                            selected_prompt = st.session_state.generated_data[st.session_state.count][f"{real_prompt_type}_prompt"]
                            st.session_state.generated_data[st.session_state.count]['sides'][(dim,op)] = side_index
                            st.session_state.generated_data[st.session_state.count]['prompts'][(dim,op)] = real_prompt_type
                st.divider()

            num_of_fully_annotated_items = len([x["prompts"] for x in st.session_state.generated_data if len(x["prompts"]) == len(dimensions)*len(options)])
            st.write(f"Annotation for {num_of_fully_annotated_items} out of {len(st.session_state.generated_data)} is completed")
            finish_clicked = st.button(f"Submit", disabled = num_of_fully_annotated_items < MIN_EXAMPLE_TO_EVALUATE)
            if finish_clicked:
                if validate_annotation():
                    # showing aggregated results
                    results, num_of_examples = calculate_results()
                    st.write(f"Submitted annotations for {num_of_examples} examples")
                    save_results()
                    #st.write(f"Compared between {len(st.session_state.eval_prompts)} prompts")
                    #for dim in dimensions:
                    #    st.write(f"{dim}:")
                    #    for prompt_type in results:
                    #        num_of_time_prompt_is_best = results[prompt_type].get((dim, "Best"), 0)
                    #        pct_val = '{0:.2f}'.format(100*num_of_time_prompt_is_best/num_of_examples)
                    #        st.write(f"{prompt_type} prompt was chosen {num_of_time_prompt_is_best} {'times' if num_of_time_prompt_is_best != 1 else 'time'} ({pct_val}%) ")
                    st.write("Your annotation is saved. Thank you for contributing to the CPE project!")

if __name__ == "__main__":
    run()
