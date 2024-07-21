import os

import pandas as pd
import re
from collections import Counter

# chat_output_path = '/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Orith_BAM/07-07-2024 13:10:27'
# chat_output_path = '/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Artem_BAM/09-07-2024 15:14:31'
chat_output_path = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/oritht/14-07-2024 12:36:46"

target_model = 'llama-3'

offline_eval_results = f'llm_judge/{target_model}/test_full.offline.llm_judge_evaluation.csv'
eval_llm_file = os.path.join(chat_output_path, offline_eval_results)

llm_eval_results = f'llm_judge/{target_model}/eval_results.chat.llm_judge_evaluation.csv'
eval_chat_llm_file = os.path.join(chat_output_path, llm_eval_results)

summary_prompt_types = ['baseline', 'zero_shot', 'few_shot']
DO_NOT_EVALUATE_FEW_SHOT = True


def get_normalized_counts(type_name, counts, with_print=True):
    c_norm = {k: v / sum(counts.values()) for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}
    if with_print:
        print(type_name, c_norm, sum(counts.values()))
    return c_norm


def analyze_llm_evaluation(df):
    print('\nAbsolute scores (mean, var):')
    for col in df.columns:
        if DO_NOT_EVALUATE_FEW_SHOT and "few_shot" in col:
            continue
        if 'llm_judge_abs_result' in col:
            print(col, f'{df[col].mean():.2f}', f'{df[col].var():.2f}')

    print('\nRelative scores:')
    total_counts = {}
    for col in df.columns:
        if DO_NOT_EVALUATE_FEW_SHOT and "few_shot" in col:
            continue
        if 'llm_judge_rel_result' in col:
            counter = Counter(df[col])
            print(col, dict(counter), 'num_samples:', sum(counter.values()))
            prompt_types = "-".join(sorted(re.findall(r'\<(.*?)\>', col)))
            if prompt_types not in total_counts:
                total_counts[prompt_types] = Counter()
            total_counts[prompt_types] += counter

    print('\nTotal relative normalized counts')
    overall_counts = Counter()
    for pt, c in total_counts.items():
        get_normalized_counts(pt, c)
        overall_counts += c
    get_normalized_counts("\nOverall best prompt", overall_counts)


def get_manual_selectiom(row):
    #print("Best:", row["ranked_prompt_('dim1', 'Best')"], "Worst:", row["ranked_prompt_('dim1', 'Worst')"])
    manual_best = row["ranked_prompt_('dim1', 'Best')"]
    manual_worst = row["ranked_prompt_('dim1', 'Worst')"]
    for sp in summary_prompt_types:
        if sp != manual_best and sp != manual_worst:
            manual_middle = sp
            break
    #print("Manual Best:", manual_best)
    #print("Manual Middle:", manual_middle)
    #print("Manual Worst:", manual_worst)
    if DO_NOT_EVALUATE_FEW_SHOT:
        if "few_shot" in manual_best:
            manual_best = manual_middle
        if "few_shot" in manual_worst:
            manual_worst = manual_middle
    #print("--> Manual Best:", manual_best)
    #print("--> Manual Middle:", manual_middle)
    #print("--> Manual Worst:", manual_worst)
    return manual_best, manual_worst


# Offline eval
print(f"\n{eval_llm_file}")
df_llm_offline = pd.read_csv(eval_llm_file)
print("Num samples", len(df_llm_offline))
analyze_llm_evaluation(df_llm_offline)

# Chat eval
print(f"\n{eval_chat_llm_file}")
df_llm_chat = pd.read_csv(eval_chat_llm_file)
print("Num samples", len(df_llm_chat))
analyze_llm_evaluation(df_llm_chat)

#print(df_llm_chat["ranked_prompt_('dim1', 'Best')"])
df_llm_chat_dict = df_llm_chat.to_dict(orient='records')
for row in df_llm_chat_dict:
    manual_best, manual_worst = get_manual_selectiom(row)
    row["ranked_prompt_('dim1', 'Best')"] = manual_best
    row["ranked_prompt_('dim1', 'Worst')"] = manual_worst

df_llm_chat = pd.DataFrame.from_dict(df_llm_chat_dict)
#print(df_llm_chat["ranked_prompt_('dim1', 'Best')"])
col = "ranked_prompt_('dim1', 'Best')"
get_normalized_counts(col, Counter(df_llm_chat[col]))
col = "ranked_prompt_('dim1', 'Worst')"
get_normalized_counts(col, Counter(df_llm_chat[col]))

llm_best_prompt = []
manual_best_prompt = []
for row in df_llm_chat.to_dict(orient='records'):

    llm_selected_prompt = []

    manual_best, _ = get_manual_selectiom(row)
    manual_best_prompt.append(manual_best)

    for k in row.keys():
        if "_llm_judge_rel_result" in k:
            if DO_NOT_EVALUATE_FEW_SHOT and "few_shot" in k:
                continue
            llm_selected_prompt.append(row[k])
    norm_counts = get_normalized_counts('Overall', Counter(llm_selected_prompt), with_print=False)
    llm_best_prompt.append(max(norm_counts, key=lambda key: norm_counts[key]))

df_llm_chat['Best_llm_judge_rel'] = llm_best_prompt
df_llm_chat['Best_manual'] = manual_best_prompt
agreement = [1 if l == m else 0 for l,m in zip(manual_best_prompt, llm_best_prompt)]
print("Agreement:", sum(agreement)/len(agreement), len(agreement))








