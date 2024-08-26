import os
import pandas as pd
import json
import random

from concurrent.futures import ThreadPoolExecutor

from genai.schema import ChatRole
from conversational_prompt_engineering.backend.chat_manager_util import ChatManagerBase
from conversational_prompt_engineering.backend.prompt_building_util import TargetModelHandler, remove_tags_from_zero_shot_prompt
from conversational_prompt_engineering.util.csv_file_utils import read_user_csv_file
from conversational_prompt_engineering.data.dataset_utils import load_dataset_mapping
from conversational_prompt_engineering.configs.config_names import load_config

## Our customized rubric definition
SUMMRIZATION_RUBRIC_ORIG = """
[Is the model response aligned with the user's instructions?]
Score 1: There is no alignment between the instruction and the model response.
Score 2: There is poor alignment between the instruction and the model response. Most of the requirements are not fulfilled.
Score 3: There is partial alignment between the instruction and the model response.
Score 4: There is good alignment between the instruction and the model response. Most of the requirements are fulfilled.
Score 5: There is full alignment between the instruction and the model response.
""".strip()

# 1.8.24 change rubric with summarization
SUMMRIZATION_RUBRIC = """
[Does the model's response align with the summary requirements provided by the user in the instruction?]
Score 1: The summary requirements do not align with the model's response. Key requests are ignored.
Score 2: The summary requirements have minimal alignment with the model's response. There are significant gaps or misinterpretations.
Score 3: The summary requirements have moderate alignment with the model's response. While the main requirements are generally covered, there are notable omissions or inaccuracies.
Score 4: The summary requirements have high alignment with the model's response, with minor deviations.
Score 5: There is full alignment between the summary requirements outlined in the instruction and the model's response.
""".strip()

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

# Modified the prompt: corrected the input ("two responses" instead of "a response") and add instruction 1.
#RELATIVE_PROMPT_WO_REF = """###Task Description:
#An instruction (might include an Input inside it), two responses to evaluate, and a score rubric representing a evaluation criteria are given.
#1. Read the two responses carefully before making any decision. The order in which the responses are presented to you should not affect your decision.
#2. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
#3. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
#4. The output format should look as follows: "(write a feedback for criteria) [RESULT] (A or B)"
#5. Please do not generate any other opening, closing, and explanations.

RELATIVE_PROMPT_WO_REF = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate, and a score rubric representing a evaluation criteria are given.
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


def get_all_pairs(l, add_reversed_pairs=True):
    pairs = [(l[i], l[j]) for i in range(len(l)) for j in range(i + 1, len(l))]
    res = []
    for a,b in pairs:
        res.append((a,b))
        if add_reversed_pairs:
            res.append((b, a))
    return res


class LlmAsAJudge(ChatManagerBase):
    def __init__(self, credentials, model, conv_id, target_model, api, email_address, output_dir, config_name) -> None:
        super().__init__(credentials, model, conv_id, target_model, api, email_address, output_dir, config_name)

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
                for result in [r.strip() for r in result.split('Response')]:
                    if result in ["A", "B"]:
                        return (result, feedback), relative_map[result]
                    if result.startswith("A is the better response") or result.startswith("A is slightly better than"):
                        return ("A", feedback), relative_map["A"]
                    if result.startswith("B is the better response") or result.startswith("B is slightly better than"):
                        return ("B", feedback), relative_map["B"]
        return outputs, "-1"

    def _generate_texts_output(self, prompt, texts, few_shot_examples=[]):
        prompt_str = TargetModelHandler().format_prompt(model=self.target_bam_client.parameters['model_id'],
                                                        prompt=prompt, texts_and_outputs=few_shot_examples)
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
        user_content = ABSOLUTE_PROMPT_WO_REF.format(instruction=prompt, response=summary, rubric=SUMMRIZATION_RUBRIC)
        self.model_chat = [{'role': ChatRole.SYSTEM, 'content': ABS_SYSTEM_PROMPT},
                           {'role': ChatRole.USER, 'content': user_content}]
        resp = self._get_assistant_response()
        res = self._parse_output(resp, "absolute")
        return res

    def _evaluate_prompt_relative(self, prompt, summary_a, summary_b, res_map):
        user_content = REL_SYSTEM_PROMPT + "\n\n" + RELATIVE_PROMPT_WO_REF.format(instruction=prompt, response_A=summary_a,
                                                                                  response_B=summary_b, rubric=SUMMRIZATION_RUBRIC)
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
            row['llm_evaluated_instruction'] = prompt
            instruction_text = prompt.format(text=row["text"])
            # mode "absolute"
            for prompt_type in summary_prompt_types:
                row[f'{prompt_type}_llm_judge_abs_feedback'], row[f'{prompt_type}_llm_judge_abs_result'] = \
                    self._evaluate_prompt_absolute(instruction_text, row[f'{prompt_type}_output'])

            # mode "relative":
            if score_relative:
                for summary_a, summary_b in get_all_pairs(summary_prompt_types):
                    pair_name = f'<{summary_a}>-<{summary_b}>'
                    row[f'{pair_name}_llm_judge_rel_feedback'], row[f'{pair_name}_llm_judge_rel_result'] = \
                        self._evaluate_prompt_relative(instruction_text, row[f"{summary_a}_output"],
                                                       row[f"{summary_b}_output"],
                                                       {'A': f'{summary_a}', 'B': f'{summary_b}'})

    def chat_results_evaluation(self, chat_csv_file, eval_out_dir, target_model, summary_prompt_types, dataset_name):
        df = pd.read_csv(chat_csv_file)
        print(f'LLM AS A JUDGE: input file: {chat_csv_file}')
        print(f'num of test samples {len(df)}')

        # select the prompt for llm-as-a-judge evaluation
        prompt_to_evaluate = remove_tags_from_zero_shot_prompt(df[f'{llm_pivot_prompt}_prompt'][0], target_model)
        # prompt_to_evaluate = prompt_to_evaluate.replace("\n\nHere is a desired output of the text:", "")
        print(f'LLM AS A JUDGE: Chat results evaluation of the zero-shot prompt:\n\n {prompt_to_evaluate}')

        # call to LLM-as-a-judge
        generated_data = df.to_dict(orient='records')
        self.evaluate_prompt(prompt_to_evaluate, generated_data, summary_prompt_types)

        assert chat_csv_file.endswith(".csv")
        out_csv_file = f"{chat_csv_file.split('/')[-1].split('.')[0]}.chat.llm_judge_evaluation.csv"
        print(f'LLM AS A JUDGE: output file: {out_csv_file}')
        out_df = pd.DataFrame(generated_data)
        out_df['dataset_name'] = dataset_name
        out_df.to_csv(os.path.join(eval_out_dir, out_csv_file), index=False)

    def offline_evaluation(self, chat_params, test_file, eval_out_dir, target_model, summary_prompt_types, dataset_name, max_samples_to_evaluate=50):

        # select the prompt fpr llm-as-a-judge evaluation: use the CPE zero-shot prompt
        prompt_instruction = chat_params['prompts'][-1]
        prompt_str = TargetModelHandler().format_prompt(model=self.target_bam_client.parameters['model_id'], prompt=prompt_instruction, texts_and_outputs=[])
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
        generated_data = []
        for i in range(len(eval_texts)):
            generated_data.append({})
        for prompt_type in summary_prompt_types:
            few_shot_examples = []
            if prompt_type.endswith('few_shot'):
                for t, s in zip(chat_params["examples"], chat_params['accepted_outputs']):
                    if s is not None:
                        few_shot_examples.append({'text': t, 'summary': s})

            p = chat_params['baseline_prompts']['user_baseline_prompt'] if prompt_type.startswith('baseline') else prompt_instruction
            prompt_str, eval_outputs = self._generate_texts_output(p, eval_texts, few_shot_examples)
            for i, (t, s) in enumerate(zip(eval_texts, eval_outputs)):
                generated_data[i].update({f"{prompt_type}_prompt": prompt_str, "text": t, f"{prompt_type}_output": s})

        # call to LLM-as-a-judge
        self.evaluate_prompt(prompt_to_evaluate, generated_data, summary_prompt_types)

        out_df = pd.DataFrame(generated_data)
        out_df['dataset_name'] = dataset_name
        out_csv_file = f"{test_file.split('/')[-1].split('.')[0]}.offline.llm_judge_evaluation.csv"
        print(f'LLM AS A JUDGE: output file: {out_csv_file}')
        out_df.to_csv(os.path.join(eval_out_dir, out_csv_file), index=False)


def evaluate_chat(dataset_name):
    # Evaluate chat results
    chat_eval_csv_file = os.path.join(chat_out_path, "eval/eval_results.csv")
    print(f'LLM AS A JUDGE: chat manual evaluation file is: {chat_eval_csv_file}')
    # Evaluate chat results
    llm_judge.chat_results_evaluation(chat_eval_csv_file, eval_out_dir, target_model, summary_prompt_types, dataset_name)


def evaluate_offline_test(dataset_name, data_split):
    # Evaluate csv test data with chat prompts
    config = load_config(config_name)
    dataset_name_to_dir = load_dataset_mapping(config)
    print(dataset_name_to_dir.get(dataset_name))
    test_data_file = ""
    if data_split in dataset_name_to_dir.get(dataset_name).keys():
        test_data_file = dataset_name_to_dir.get(dataset_name)[data_split]
    if not os.path.isfile(test_data_file):
        print(f'Skip {test_data_file}, file does not exist')
        return
    print(f'LLM AS A JUDGE: dataset file is: {test_data_file}')
    # Evaluate full test with chat prompts
    llm_judge.offline_evaluation(chat_params, test_data_file, eval_out_dir, target_model, offline_summary_prompt_types, dataset_name)


if __name__ == "__main__":

    api = "bam"  # select the API "bam"/"watsonx"
    summary_prompt_types = ['baseline', 'zero_shot', 'few_shot']   # select the summaries for evaluation
    offline_test_splits = ["eval", "eval_llm"]
    offline_summary_prompt_types = summary_prompt_types + ['baseline_few_shot']
    llm_pivot_prompt = 'zero_shot' # 'few_shot' #

    print("LLM AS A JUDGE: pivot prompt", llm_pivot_prompt)

    # Chats for analysis
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_24_7_2024"
    chats_list = [
        "oritht/14-07-2024 12:36:46",
        "liat/21-07-2024 12:16:37",
        "shai/21-07-2024 12:36:52",
        "shai/wiki_animals",
    ]

    chats_list = [
        "Shai_20ng_space/24-07-2024 12:33:50",
        "Artem_cfpb/24-07-2024 10:25:30",
        "Artem_financial_news/24-07-2024 11:09:44",
        "Artem_reddit/24-07-2024 09:45:58",
        "CIO/24-07-2024 14:12:09",
        "Artem_speeches/24-07-2024 13:09:34",
        "Liat_speeches/24-07-2024 16:47:16",
        "Liat_wiki_movies/24-07-2024 17:54:36",
        "Orith_wiki_movies/25-07-2024 11:52:11",
    ]

    ## Evaluation for paper: CIO
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_CIO"
    chats_list = [
        "gmelino_microsoft/24-07-2024 14:17:00"
    ]

    ## Evaluation for paper: ISRL
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_ISRL"
    chats_list = [
        #"eladv_wiki_movies/25-07-2024 13:22:07",
        "Roi.Cohen_wiki_animals/25-07-2024 12:38:25"
    ]

    # Chats for analysis
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_24_7_2024"
    chats_list = [
        #"Shai_20ng_space/24-07-2024 12:33:50",
        #"Liat_speeches/24-07-2024 16:47:16",
        "Liat_wiki_movies/24-07-2024 17:54:36",
    ]

    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_old"
    chats_list = [
        "liat/21-07-2024 12:16:37",
    ]

    #chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_30_7_2024"
    #chats_list = [
    #    #"Ariel.gera1/30-07-2024 14:44:27",
    #    "shachar.don-yehiya/30-07-2024 10:25:36",
    #]

    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_31_7_2024"
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_1_8_2024"
    chats_list = [
        #"ella.rabinovich1_tldr/31-07-2024 09_38_05",
        #"lilache_debate_speeches/31-07-2024 07_55_01",
        #"ronicon_wiki_movies/31-07-2024 15:44:31",
        #"ronicon_wiki_animals/31-07-2024 14:45:15",
        #"matano_wiki_movies/31-07-2024 14:03:59", # dummy labels
        #"Roi.Cohen_wiki_animals/31-07-2024 10:22:19",
        #"yoavka_wiki_animals/01-08-2024 05_54_11",
        #"noams/30-07-2024 10:49:32",
        "koren.lazar_wiki_animals/01-08-2024 13:43:30",
    ]

    # Credentials for API
    if api == "bam":
        credentials = {"key": os.environ["BAM_APIKEY"]}
    elif api == "watsonx":
        credentials = {"key": os.environ["WATSONX_APIKEY"], "project_id": os.environ["PROJECT_ID"]}
    else:
        credentials = {}

    for chat_dir in chats_list:
        chat_out_path = os.path.join(chats_output_dir, chat_dir)
        print(f"Evaluating {chat_dir}")

        # the chat result json file
        chat_res_json_file = "chat_result.json"
        # load the information - and extract relevant info
        with open(os.path.join(chat_out_path, chat_res_json_file), "r") as f:
            chat_params = json.load(f)

        # config name
        config_name = chat_params["config_name"]
        print(f'LLM AS A JUDGE: config name: {config_name}')

        # the dataset to evaluate
        dataset_name = chat_params["dataset_name"]
        print(f'LLM AS A JUDGE: dataset is: {dataset_name}')

        # the model that generates the summaries
        target_model = get_model_id(chat_params["target_model"])
        print(f'LLM AS A JUDGE: target_model is: {target_model}')

        #eval_out_dir = os.path.join(chat_out_path, f"llm_judge/pivot_{llm_pivot_prompt}/{target_model}")
        # 1.8.24 change rubric with summarization
        eval_out_dir = os.path.join(chat_out_path, f"llm_judge_new_rubric/pivot_{llm_pivot_prompt}/{target_model}")
        os.makedirs(eval_out_dir, exist_ok=True)
        print(f'LLM AS A JUDGE: output directory: {eval_out_dir}')

        llm_judge = LlmAsAJudge(credentials=credentials, model="prometheus_7b",
                                conv_id="llm_as_a_judge_offline", target_model=target_model, api=api, email_address=os.environ["IBM_EMAIL"],
                                output_dir=chat_out_path, config_name=config_name)

        # LLM Evaluation on chat
        evaluate_chat(dataset_name)
        # LLM Evaluation on offline test data
        for split in offline_test_splits:
            evaluate_offline_test(split)


