import os
import pandas as pd

from genai.schema import ChatRole
from conversational_prompt_engineering.backend.chat_manager_util import ChatManagerBase

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


class LlmAsAJudge(ChatManagerBase):
    def __init__(self, bam_api_key, model, conv_id) -> None:
        super().__init__(bam_api_key, model, conv_id)

        self.model_chat = []

    def _get_assistant_response(self, chat=None, max_new_tokens=None):
        return super()._get_assistant_response(chat or self.model_chat, max_new_tokens)

    def _parse_output(self, outputs, mode: str):
        parts = outputs.split("[RESULT]")
        if len(parts) == 2:
            feedback, result = parts[0].strip(), parts[1].strip()
            if mode == "absolute":
                if result.isdigit() and result in ["1", "2", "3", "4", "5"]:
                    return feedback, result
            elif mode == "relative":
                if result in ["A", "B"]:
                    return feedback, result
        return "", "-1"

    def _evaluate_prompt_absolute(self, prompt, summary):
        user_content = ABSOLUTE_PROMPT_WO_REF.format(instruction=prompt, response=summary, rubric=HELPFULNESS_RUBRIC)
        self.model_chat = [{'role': ChatRole.SYSTEM, 'content': ABS_SYSTEM_PROMPT},
                           {'role': ChatRole.USER, 'content': user_content}]
        resp = self._get_assistant_response()
        res = self._parse_output(resp, "absolute")
        return res

    def _evaluate_prompt_relative(self, prompt, summary_a, summary_b):
        user_content = REL_SYSTEM_PROMPT + "\n\n" + RELATIVE_PROMPT_WO_REF.format(instruction=prompt, response_A=summary_a,
                                                                                  response_B=summary_b, rubric=HELPFULNESS_RUBRIC)
        self.model_chat = [{'role': ChatRole.SYSTEM, 'content': REL_SYSTEM_PROMPT},
                           {'role': ChatRole.USER, 'content': user_content}]
        resp = self._get_assistant_response()
        res = self._parse_output(resp, "relative")
        return res

    def evaluate_prompt(self, prompt, summaries):
        for row in summaries:
            row['llm_judge'] = {}
            instruction_text = prompt.format(text=row["text"])
            # mode "absolute"
            row['llm_judge'].update(
                [(f'baseline_llm_judge_abs', self._evaluate_prompt_absolute(instruction_text, row[f'baseline_summary']))])
            row['llm_judge'].update(
                [(f'zero_shot_llm_judge_abs', self._evaluate_prompt_absolute(instruction_text, row[f'zero_shot_summary']))])
            row['llm_judge'].update(
                [(f'few_shot_llm_judge_abs', self._evaluate_prompt_absolute(instruction_text, row[f'few_shot_summary']))])
            # mode "relative":
            row['llm_judge'].update([(f'BL_FS_llm_judge_rel', self._evaluate_prompt_relative(instruction_text, row["baseline_summary"], row["few_shot_summary"]))])
            row['llm_judge'].update([(f'BL_ZS_llm_judge_rel', self._evaluate_prompt_relative(instruction_text, row["baseline_summary"], row["zero_shot_summary"]))])
            row['llm_judge'].update([(f'ZS_FS_llm_judge_rel', self._evaluate_prompt_relative(instruction_text, row["zero_shot_summary"], row["few_shot_summary"]))])


if __name__ == "__main__":

    api_key = os.environ['BAM_APIKEY']

    chat_csv_path = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/a0a8d57d8602e844/30-06-2024 10:56:26/eval"
    chat_csv_file = "eval_results.csv"

    llm_judge = LlmAsAJudge(bam_api_key=api_key, model="prometheus_7b",
                            conv_id="llm_as_a_judge_offline")

    df = pd.read_csv(os.path.join(chat_csv_path, chat_csv_file))
    prompt = df["zero_shot_prompt"][0]
    print(f'LLM AS A JUDGE: evaluating zero-shot prompt:\n\n {prompt}')

    generated_data = df.to_dict(orient='records')
    llm_judge.evaluate_prompt(prompt, generated_data)

    out_csv_file = "eval_results_with_llm_judge.csv"
    out_df = pd.DataFrame(generated_data)
    out_df.to_csv(os.path.join(chat_csv_path, out_csv_file))
