import os
import pandas as pd
import json
import random

from genai.schema import ChatRole
from conversational_prompt_engineering.backend.chat_manager_util import ChatManagerBase
from conversational_prompt_engineering.backend.prompt_building_util import build_few_shot_prompt, remove_tags_from_zero_shot_prompt
from conversational_prompt_engineering.util.csv_file_utils import read_user_csv_file
from conversational_prompt_engineering.data.dataset_name_to_dir import dataset_name_to_dir
from concurrent.futures import ThreadPoolExecutor

## Prompts definitions are taken from: https://github.com/prometheus-eval/prometheus-eval/blob/main/libs/prometheus-eval/prometheus_eval/prompts.py

HELPFULNESS_RUBRIC = """
[Does the model provide relevant and useful responses to the user's needs or questions?]
Score 1: The model’s responses are irrelevant or unhelpful to the user's needs or queries.
Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries.
""".strip()

ABS_SYSTEM_PROMPT = "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
REL_SYSTEM_PROMPT = "You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort."

ABSOLUTE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

RELATIVE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_A}

###Response B:
{response_B}

###Score Rubric:
{rubric}

###Feedback: """


def get_model_id(model_name):
    if "llama-3" in model_name:
        return "llama-3"
    elif "mixtral" in model_name:
        return "mixtral"
    elif "granite" in model_name:
        return "granite"
    else:
        print("Unsupported model")
    return None


class LlmAsAJudge(ChatManagerBase):
    def __init__(self, credentials, model, conv_id, target_model, api, email_address) -> None:
        super().__init__(credentials, model, conv_id, target_model, api, email_address)

        self.model_chat = []

    def _get_assistant_response(self, chat=None, max_new_tokens=None):
        return super()._get_assistant_response(chat or self.model_chat, max_new_tokens)

    def _parse_output(self, outputs, mode: str, relative_map=None):
        parts = outputs.split("[RESULT]")
        if len(parts) == 2:
            feedback, result = parts[0].strip(), parts[1].strip()
            if mode == "absolute":
                if result.isdigit() and result in ["1", "2", "3", "4", "5"]:
                    return feedback, result
            elif mode == "relative":
                result = result.split('Response')[-1].strip()
                if result in ["A", "B"]:
                    return (result, feedback), relative_map[result]
        return outputs, "-1"

    def _generate_texts_output(self, prompt, texts, few_shot_examples=[]):
        prompt_str = build_few_shot_prompt(prompt,
                                           few_shot_examples,
                                           self.target_bam_client.parameters['model_id'])
        futures = {}
        with ThreadPoolExecutor(max_workers=len(texts)) as executor:
            for i, example in enumerate(texts):
                prompt_text = prompt_str.format(text=example)
                futures[i] = executor.submit(self._generate_output, prompt_text)
        outputs = []
        for _, f in futures.items():
            outputs.append(f.result())
        return prompt_str, outputs

    def _evaluate_prompt_absolute(self, prompt, summary):
        user_content = ABSOLUTE_PROMPT_WO_REF.format(instruction=prompt, response=summary, rubric=HELPFULNESS_RUBRIC)
        self.model_chat = [{'role': ChatRole.SYSTEM, 'content': ABS_SYSTEM_PROMPT},
                           {'role': ChatRole.USER, 'content': user_content}]
        resp = self._get_assistant_response()
        res = self._parse_output(resp, "absolute")
        return res

    def _evaluate_prompt_relative(self, prompt, summary_a, summary_b, res_map):
        user_content = REL_SYSTEM_PROMPT + "\n\n" + RELATIVE_PROMPT_WO_REF.format(instruction=prompt, response_A=summary_a,
                                                                                  response_B=summary_b, rubric=HELPFULNESS_RUBRIC)
        self.model_chat = [{'role': ChatRole.SYSTEM, 'content': REL_SYSTEM_PROMPT},
                           {'role': ChatRole.USER, 'content': user_content}]
        resp = self._get_assistant_response()
        res = self._parse_output(resp, "relative", res_map)
        return res

    def evaluate_prompt(self, prompt, summaries, summary_prompt_types, score_relative=True):
        idx = 0
        for row in summaries:
            idx += 1
            print(f'evaluate sample {idx}...')
            instruction_text = prompt.format(text=row["text"])
            # mode "absolute"
            for prompt_type in summary_prompt_types:
                row[f'{prompt_type}_llm_judge_abs_feedback'], row[f'{prompt_type}_llm_judge_abs_result'] = \
                    self._evaluate_prompt_absolute(instruction_text, row[f'{prompt_type}_summary'])

            # mode "relative":
            if score_relative:
                row['BL_FS_llm_judge_rel_feedback'], row['BL_FS_llm_judge_rel_result'] = \
                    self._evaluate_prompt_relative(instruction_text, row["baseline_summary"], row["few_shot_summary"],
                                                   {'A': 'baseline', 'B': 'few_shot'})
                row['BL_ZS_llm_judge_rel_feedback'], row['BL_ZS_llm_judge_rel_result'] = \
                    self._evaluate_prompt_relative(instruction_text, row["baseline_summary"], row["zero_shot_summary"],
                                                   {'A': 'baseline', 'B': 'zero_shot'})
                row['ZS_FS_llm_judge_rel_feedback'], row['ZS_FS_llm_judge_rel_result'] = \
                    self._evaluate_prompt_relative(instruction_text, row["zero_shot_summary"], row["few_shot_summary"],
                                                   {'A': 'zero_shot', 'B': 'few_shot'})

    def chat_results_evaluation(self, chat_csv_file, eval_out_dir, target_model):
        df = pd.read_csv(chat_csv_file)
        print(f'LLM AS A JUDGE: input file: {chat_csv_file}')
        print(f'num of test samples {len(df)}')

        # select the prompt for llm-as-a-judge evaluation
        prompt_to_evaluate = remove_tags_from_zero_shot_prompt(df["zero_shot_prompt"][0], target_model)
        print(f'LLM AS A JUDGE: Chat results evaluation of the zero-shot prompt:\n\n {prompt_to_evaluate}')

        # call to LLM-as-a-judge
        summary_prompt_types = ['baseline', 'zero_shot', 'few_shot']
        generated_data = df.to_dict(orient='records')
        self.evaluate_prompt(prompt_to_evaluate, generated_data, summary_prompt_types)

        assert chat_csv_file.endswith(".csv")
        out_csv_file = f"{chat_csv_file.split('/')[-1].split('.')[0]}.chat.llm_judge_evaluation.csv"
        print(f'LLM AS A JUDGE: output file: {out_csv_file}')
        out_df = pd.DataFrame(generated_data)
        out_df.to_csv(os.path.join(eval_out_dir, out_csv_file))

    def offline_evaluation(self, chat_params, test_file, eval_out_dir, target_model, max_samples_to_evaluate=50):

        # select the prompt fpr llm-as-a-judge evaluation: use the CPE zero-shot prompt
        prompt_str = build_few_shot_prompt(chat_params['prompts'][-1], [], self.target_bam_client.parameters['model_id'])
        prompt_to_evaluate = remove_tags_from_zero_shot_prompt(prompt_str, target_model)
        print(f'LLM AS A JUDGE: Offline evaluation of the CPE zero-shot prompt:\n\n {prompt_to_evaluate}')

        # upload the test data
        eval_texts = read_user_csv_file(test_file).text.tolist()
        print(f'num of test samples {len(eval_texts)}')
        # select a subset of the data in random for evaluation
        if len(eval_texts) > max_samples_to_evaluate:
            random.seed(32)
            random.shuffle(eval_texts)
            print('shuffle input texts')
        eval_texts = eval_texts[:max_samples_to_evaluate]
        print(f'num of test samples to evaluate {len(eval_texts)}')

        # generate summaries
        summary_prompt_types = ['baseline', 'zero_shot', 'few_shot']
        generated_data = []
        for i in range(len(eval_texts)):
            generated_data.append({})
        for prompt_type in summary_prompt_types:
            few_shot_examples = []
            if prompt_type == 'few_shot':
                for t, s in zip(chat_params["examples"], chat_params['accepted_outputs']):
                    if s is not None:
                        few_shot_examples.append({'text': t, 'summary': s})

            p = chat_params['baseline_prompts']['model_baseline_prompt'] if prompt_type == 'baseline' else prompt_to_evaluate
            prompt_str, eval_outputs = self._generate_texts_output(p, eval_texts, few_shot_examples)
            for i, (t, s) in enumerate(zip(eval_texts, eval_outputs)):
                generated_data[i].update({f"{prompt_type}_prompt": prompt_str, "text": t, f"{prompt_type}_summary": s})

        # call to LLM-as-a-judge
        self.evaluate_prompt(prompt_to_evaluate, generated_data, summary_prompt_types)

        out_df = pd.DataFrame(generated_data)
        out_csv_file = f"{test_file.split('/')[-1].split('.')[0]}.offline.llm_judge_evaluation.csv"
        print(f'LLM AS A JUDGE: output file: {out_csv_file}')
        out_df.to_csv(os.path.join(eval_out_dir, out_csv_file))


if __name__ == "__main__":

    api = "bam"  # select the API "bam"/"watsonx"
    evaluation_mode = "chat_eval"  # select the evaluation mode "chat_eval"/"test_csv"
    evaluation_data_split = "eval"  # select the dataset split csv to evaluate (when evaluation mode is "test_csv")

    #chat_out_path = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Orith_BAM/07-07-2024 13:10:27"
    chat_out_path = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Artem_BAM/09-07-2024 15:14:31"

    chat_res_json_file = "chat_result.json"
    # load the information - and extract relevant info
    with open(os.path.join(chat_out_path, chat_res_json_file), "r") as f:
        chat_params = json.load(f)

    # the dataset to evaluate
    dataset_name = chat_params["dataset_name"]
    print(f'LLM AS A JUDGE: dataset is {dataset_name} split {evaluation_data_split}')

    # the model that generates the summaries
    target_model = get_model_id(chat_params["target_model"])
    print(f'LLM AS A JUDGE: target_model is {target_model}')

    eval_out_dir = os.path.join(chat_out_path, f"llm_judge/{target_model}")
    os.makedirs(eval_out_dir, exist_ok=True)
    print(f'LLM AS A JUDGE: output directory: {eval_out_dir}')

    if evaluation_mode == "chat_eval":
        # Evaluate chat results
        chat_eval_csv_file = os.path.join(chat_out_path, "eval/eval_results.csv")
        print(f'LLM AS A JUDGE: chat manual evaluation file is {chat_eval_csv_file}')
    elif evaluation_mode == "test_csv":
        # Evaluate csv test data with chat prompts
        test_data_file = dataset_name_to_dir.get(dataset_name)[evaluation_data_split]
        print(f'LLM AS A JUDGE: dataset file is {test_data_file}')
    else:
        print(f"Wrong evaluation mode {evaluation_mode}")
        exit()

    # Credentials for API
    if api == "bam":
        credentials = {"key": os.environ["BAM_APIKEY"]}
    elif api == "watsonx":
        credentials = {"key": os.environ["WATSONX_APIKEY"], "project_id": os.environ["PROJECT_ID"]}
    else:
        credentials = {}

    llm_judge = LlmAsAJudge(credentials=credentials, model="prometheus_7b",
                            conv_id="llm_as_a_judge_offline", target_model=target_model, api=api, email_address=os.environ["IBM_EMAIL"])

    if evaluation_mode == "chat_eval":
        # Evaluate chat results
        llm_judge.chat_results_evaluation(chat_eval_csv_file, eval_out_dir, target_model)
    elif evaluation_mode == "test_csv":
        # Evaluate full test with chat prompts
        llm_judge.offline_evaluation(chat_params, test_data_file, eval_out_dir, target_model)
    else:
        pass
